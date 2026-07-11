use crate::audio_toolkit::{apply_custom_words, filter_transcription_output};
use crate::managers::audio::AudioRecordingManager;
use crate::managers::model::{EngineType, ModelManager};
use crate::settings::{
    get_settings, ModelUnloadTimeout, OrtAcceleratorSetting, WhisperAcceleratorSetting,
};
use anyhow::Result;
use log::{debug, error, info, warn};
use serde::Serialize;
use specta::Type;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::thread;
use std::time::{Duration, SystemTime};
use tauri::{AppHandle, Emitter, Manager};
use transcribe_rs::{
    onnx::{
        canary::CanaryModel,
        gigaam::GigaAMModel,
        moonshine::{MoonshineModel, MoonshineVariant, StreamingModel},
        parakeet::{ParakeetModel, ParakeetParams, TimestampGranularity},
        sense_voice::{SenseVoiceModel, SenseVoiceParams},
        Quantization,
    },
    whisper_cpp::{WhisperEngine, WhisperInferenceParams},
    SpeechModel, TranscribeOptions,
};

#[derive(Clone, Debug, Serialize)]
pub struct ModelStateEvent {
    pub event_type: String,
    pub model_id: Option<String>,
    pub model_name: Option<String>,
    pub error: Option<String>,
}

enum LoadedEngine {
    Whisper(WhisperEngine),
    Parakeet(ParakeetModel),
    Moonshine(MoonshineModel),
    MoonshineStreaming(StreamingModel),
    SenseVoice(SenseVoiceModel),
    GigaAM(GigaAMModel),
    Canary(CanaryModel),
    GroqWhisper,
}

/// RAII guard that clears the `is_loading` flag and notifies waiters on drop.
/// Ensures the loading flag is always reset, even on early returns or panics.
pub struct LoadingGuard {
    is_loading: Arc<Mutex<bool>>,
    loading_condvar: Arc<Condvar>,
}

impl Drop for LoadingGuard {
    fn drop(&mut self) {
        let mut is_loading = self.is_loading.lock().unwrap();
        *is_loading = false;
        self.loading_condvar.notify_all();
    }
}

#[derive(Clone)]
pub struct TranscriptionManager {
    engine: Arc<Mutex<Option<LoadedEngine>>>,
    model_manager: Arc<ModelManager>,
    app_handle: AppHandle,
    current_model_id: Arc<Mutex<Option<String>>>,
    last_activity: Arc<AtomicU64>,
    shutdown_signal: Arc<AtomicBool>,
    watcher_handle: Arc<Mutex<Option<thread::JoinHandle<()>>>>,
    is_loading: Arc<Mutex<bool>>,
    loading_condvar: Arc<Condvar>,
}

impl TranscriptionManager {
    pub fn new(app_handle: &AppHandle, model_manager: Arc<ModelManager>) -> Result<Self> {
        let manager = Self {
            engine: Arc::new(Mutex::new(None)),
            model_manager,
            app_handle: app_handle.clone(),
            current_model_id: Arc::new(Mutex::new(None)),
            last_activity: Arc::new(AtomicU64::new(Self::now_ms())),
            shutdown_signal: Arc::new(AtomicBool::new(false)),
            watcher_handle: Arc::new(Mutex::new(None)),
            is_loading: Arc::new(Mutex::new(false)),
            loading_condvar: Arc::new(Condvar::new()),
        };

        // Start the idle watcher
        {
            let app_handle_cloned = app_handle.clone();
            let manager_cloned = manager.clone();
            let shutdown_signal = manager.shutdown_signal.clone();
            let handle = thread::spawn(move || {
                debug!("Idle watcher thread started");
                while !shutdown_signal.load(Ordering::Relaxed) {
                    thread::sleep(Duration::from_secs(10)); // Check every 10 seconds

                    // Check shutdown signal again after sleep
                    if shutdown_signal.load(Ordering::Relaxed) {
                        break;
                    }

                    let settings = get_settings(&app_handle_cloned);
                    let timeout = settings.model_unload_timeout;

                    // Skip Immediately — that variant is handled by
                    // maybe_unload_immediately() after each transcription.
                    // Treating it as 0s here would unload the model mid-recording.
                    if timeout == ModelUnloadTimeout::Immediately {
                        continue;
                    }

                    // While recording, keep the idle timer fresh so the
                    // model is never unloaded mid-session.
                    let is_recording = app_handle_cloned
                        .try_state::<Arc<AudioRecordingManager>>()
                        .map_or(false, |a| a.is_recording());
                    if is_recording {
                        manager_cloned.touch_activity();
                        continue;
                    }

                    if let Some(limit_seconds) = timeout.to_seconds() {
                        let last = manager_cloned.last_activity.load(Ordering::Relaxed);
                        let now_ms = TranscriptionManager::now_ms();
                        let idle_ms = now_ms.saturating_sub(last);
                        let limit_ms = limit_seconds * 1000;

                        if idle_ms > limit_ms {
                            // idle -> unload
                            if manager_cloned.is_model_loaded() {
                                let unload_start = std::time::Instant::now();
                                info!(
                                    "Model idle for {}s (limit: {}s), unloading",
                                    idle_ms / 1000,
                                    limit_seconds
                                );
                                match manager_cloned.unload_model() {
                                    Ok(()) => {
                                        let unload_duration = unload_start.elapsed();
                                        info!(
                                            "Model unloaded due to inactivity (took {}ms)",
                                            unload_duration.as_millis()
                                        );
                                    }
                                    Err(e) => {
                                        error!("Failed to unload idle model: {}", e);
                                    }
                                }
                            }
                        }
                    }
                }
                debug!("Idle watcher thread shutting down gracefully");
            });
            *manager.watcher_handle.lock().unwrap() = Some(handle);
        }

        Ok(manager)
    }

    /// Lock the engine mutex, recovering from poison if a previous transcription panicked.
    fn lock_engine(&self) -> MutexGuard<'_, Option<LoadedEngine>> {
        self.engine.lock().unwrap_or_else(|poisoned| {
            warn!("Engine mutex was poisoned by a previous panic, recovering");
            poisoned.into_inner()
        })
    }

    pub fn is_model_loaded(&self) -> bool {
        let engine = self.lock_engine();
        engine.is_some()
    }

    /// Atomically check whether a model load is in progress and, if not, mark
    /// one as starting. Returns a [`LoadingGuard`] whose [`Drop`] impl will
    /// clear the flag and wake waiters. Returns `None` if a load is already in
    /// progress.
    pub fn try_start_loading(&self) -> Option<LoadingGuard> {
        let mut is_loading = self.is_loading.lock().unwrap();
        if *is_loading {
            return None;
        }
        *is_loading = true;
        Some(LoadingGuard {
            is_loading: self.is_loading.clone(),
            loading_condvar: self.loading_condvar.clone(),
        })
    }

    pub fn unload_model(&self) -> Result<()> {
        let unload_start = std::time::Instant::now();
        debug!("Starting to unload model");

        {
            let mut engine = self.lock_engine();
            // Dropping the engine frees all resources
            *engine = None;
        }
        {
            let mut current_model = self.current_model_id.lock().unwrap();
            *current_model = None;
        }

        // Emit unloaded event
        let _ = self.app_handle.emit(
            "model-state-changed",
            ModelStateEvent {
                event_type: "unloaded".to_string(),
                model_id: None,
                model_name: None,
                error: None,
            },
        );

        let unload_duration = unload_start.elapsed();
        debug!(
            "Model unloaded manually (took {}ms)",
            unload_duration.as_millis()
        );
        Ok(())
    }

    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }

    /// Reset the idle timer to now.
    fn touch_activity(&self) {
        self.last_activity.store(Self::now_ms(), Ordering::Relaxed);
    }

    /// Unloads the model immediately if the setting is enabled and the model is loaded
    pub fn maybe_unload_immediately(&self, context: &str) {
        let settings = get_settings(&self.app_handle);
        if settings.model_unload_timeout == ModelUnloadTimeout::Immediately
            && self.is_model_loaded()
        {
            info!("Immediately unloading model after {}", context);
            if let Err(e) = self.unload_model() {
                warn!("Failed to immediately unload model: {}", e);
            }
        }
    }

    pub fn load_model(&self, model_id: &str) -> Result<()> {
        let load_start = std::time::Instant::now();
        debug!("Starting to load model: {}", model_id);

        // Emit loading started event
        let _ = self.app_handle.emit(
            "model-state-changed",
            ModelStateEvent {
                event_type: "loading_started".to_string(),
                model_id: Some(model_id.to_string()),
                model_name: None,
                error: None,
            },
        );

        let model_info = self
            .model_manager
            .get_model_info(model_id)
            .ok_or_else(|| anyhow::anyhow!("Model not found: {}", model_id))?;

        if !model_info.is_downloaded {
            let error_msg = "Model not downloaded";
            let _ = self.app_handle.emit(
                "model-state-changed",
                ModelStateEvent {
                    event_type: "loading_failed".to_string(),
                    model_id: Some(model_id.to_string()),
                    model_name: Some(model_info.name.clone()),
                    error: Some(error_msg.to_string()),
                },
            );
            return Err(anyhow::anyhow!(error_msg));
        }

        // GroqWhisper is API-based — no local model file to load
        if matches!(model_info.engine_type, EngineType::GroqWhisper) {
            let mut engine_guard = self.lock_engine();
            *engine_guard = Some(LoadedEngine::GroqWhisper);
            drop(engine_guard);
            {
                let mut current_model = self.current_model_id.lock().unwrap();
                *current_model = Some(model_id.to_string());
            }
            let _ = self.app_handle.emit(
                "model-state-changed",
                ModelStateEvent {
                    event_type: "loaded".to_string(),
                    model_id: Some(model_id.to_string()),
                    model_name: Some(model_info.name.clone()),
                    error: None,
                },
            );
            return Ok(());
        }

        let model_path = self.model_manager.get_model_path(model_id)?;

        // Create appropriate engine based on model type
        let emit_loading_failed = |error_msg: &str| {
            let _ = self.app_handle.emit(
                "model-state-changed",
                ModelStateEvent {
                    event_type: "loading_failed".to_string(),
                    model_id: Some(model_id.to_string()),
                    model_name: Some(model_info.name.clone()),
                    error: Some(error_msg.to_string()),
                },
            );
        };

        let loaded_engine = match model_info.engine_type {
            EngineType::Whisper => {
                let engine = WhisperEngine::load(&model_path).map_err(|e| {
                    let error_msg = format!("Failed to load whisper model {}: {}", model_id, e);
                    emit_loading_failed(&error_msg);
                    anyhow::anyhow!(error_msg)
                })?;
                LoadedEngine::Whisper(engine)
            }
            EngineType::Parakeet => {
                let engine =
                    ParakeetModel::load(&model_path, &Quantization::Int8).map_err(|e| {
                        let error_msg =
                            format!("Failed to load parakeet model {}: {}", model_id, e);
                        emit_loading_failed(&error_msg);
                        anyhow::anyhow!(error_msg)
                    })?;
                LoadedEngine::Parakeet(engine)
            }
            EngineType::Moonshine => {
                let engine = MoonshineModel::load(
                    &model_path,
                    MoonshineVariant::Base,
                    &Quantization::default(),
                )
                .map_err(|e| {
                    let error_msg = format!("Failed to load moonshine model {}: {}", model_id, e);
                    emit_loading_failed(&error_msg);
                    anyhow::anyhow!(error_msg)
                })?;
                LoadedEngine::Moonshine(engine)
            }
            EngineType::MoonshineStreaming => {
                let engine = StreamingModel::load(&model_path, 0, &Quantization::default())
                    .map_err(|e| {
                        let error_msg = format!(
                            "Failed to load moonshine streaming model {}: {}",
                            model_id, e
                        );
                        emit_loading_failed(&error_msg);
                        anyhow::anyhow!(error_msg)
                    })?;
                LoadedEngine::MoonshineStreaming(engine)
            }
            EngineType::SenseVoice => {
                let engine =
                    SenseVoiceModel::load(&model_path, &Quantization::Int8).map_err(|e| {
                        let error_msg =
                            format!("Failed to load SenseVoice model {}: {}", model_id, e);
                        emit_loading_failed(&error_msg);
                        anyhow::anyhow!(error_msg)
                    })?;
                LoadedEngine::SenseVoice(engine)
            }
            EngineType::GigaAM => {
                let engine = GigaAMModel::load(&model_path, &Quantization::Int8).map_err(|e| {
                    let error_msg = format!("Failed to load gigaam model {}: {}", model_id, e);
                    emit_loading_failed(&error_msg);
                    anyhow::anyhow!(error_msg)
                })?;
                LoadedEngine::GigaAM(engine)
            }
            EngineType::Canary => {
                let engine = CanaryModel::load(&model_path, &Quantization::Int8).map_err(|e| {
                    let error_msg = format!("Failed to load canary model {}: {}", model_id, e);
                    emit_loading_failed(&error_msg);
                    anyhow::anyhow!(error_msg)
                })?;
                LoadedEngine::Canary(engine)
            }
            EngineType::GroqWhisper => unreachable!("GroqWhisper handled above"),
        };

        // Update the current engine and model ID
        {
            let mut engine = self.lock_engine();
            *engine = Some(loaded_engine);
        }
        {
            let mut current_model = self.current_model_id.lock().unwrap();
            *current_model = Some(model_id.to_string());
        }

        // Reset idle timer so the watcher doesn't immediately unload a just-loaded model
        self.touch_activity();

        // Emit loading completed event
        let _ = self.app_handle.emit(
            "model-state-changed",
            ModelStateEvent {
                event_type: "loading_completed".to_string(),
                model_id: Some(model_id.to_string()),
                model_name: Some(model_info.name.clone()),
                error: None,
            },
        );

        let load_duration = load_start.elapsed();
        debug!(
            "Successfully loaded transcription model: {} (took {}ms)",
            model_id,
            load_duration.as_millis()
        );
        Ok(())
    }

    /// Kicks off the model loading in a background thread if it's not already loaded
    pub fn initiate_model_load(&self) {
        let mut is_loading = self.is_loading.lock().unwrap();
        if *is_loading || self.is_model_loaded() {
            return;
        }

        *is_loading = true;
        let self_clone = self.clone();
        thread::spawn(move || {
            let settings = get_settings(&self_clone.app_handle);
            if let Err(e) = self_clone.load_model(&settings.selected_model) {
                error!("Failed to load model: {}", e);
            }
            let mut is_loading = self_clone.is_loading.lock().unwrap();
            *is_loading = false;
            self_clone.loading_condvar.notify_all();
        });
    }

    pub fn get_current_model(&self) -> Option<String> {
        let current_model = self.current_model_id.lock().unwrap();
        current_model.clone()
    }

    pub fn transcribe(&self, audio: Vec<f32>, wav_hint: Option<std::path::PathBuf>) -> Result<String> {
        #[cfg(debug_assertions)]
        if std::env::var("HANDY_FORCE_TRANSCRIPTION_FAILURE").is_ok() {
            return Err(anyhow::anyhow!(
                "Simulated transcription failure (HANDY_FORCE_TRANSCRIPTION_FAILURE)"
            ));
        }

        // Update last activity timestamp
        self.touch_activity();

        let st = std::time::Instant::now();

        debug!("Audio vector length: {}", audio.len());

        if audio.is_empty() {
            debug!("Empty audio vector");
            self.maybe_unload_immediately("empty audio");
            return Ok(String::new());
        }

        // Check if model is loaded, if not try to load it
        {
            // If the model is loading, wait for it to complete.
            let mut is_loading = self.is_loading.lock().unwrap();
            while *is_loading {
                is_loading = self.loading_condvar.wait(is_loading).unwrap();
            }

            let engine_guard = self.lock_engine();
            if engine_guard.is_none() {
                return Err(anyhow::anyhow!("Model is not loaded for transcription."));
            }
        }

        // Get current settings for configuration
        let settings = get_settings(&self.app_handle);

        // Validate selected language against the model's supported languages.
        // If the language isn't supported, fall back to "auto" to prevent errors.
        let validated_language = if settings.selected_language == "auto" {
            "auto".to_string()
        } else {
            let is_supported = self
                .model_manager
                .get_model_info(&settings.selected_model)
                .map(|info| {
                    info.supported_languages.is_empty()
                        || info
                            .supported_languages
                            .contains(&settings.selected_language)
                })
                .unwrap_or(true);

            if is_supported {
                settings.selected_language.clone()
            } else {
                warn!(
                    "Language '{}' not supported by current model, falling back to auto-detect",
                    settings.selected_language
                );
                "auto".to_string()
            }
        };

        // Perform transcription with the appropriate engine.
        // We use catch_unwind to prevent engine panics from poisoning the mutex,
        // which would make the app hang indefinitely on subsequent operations.
        let result = {
            let mut engine_guard = self.lock_engine();

            // Take the engine out so we own it during transcription.
            // If the engine panics, we simply don't put it back (effectively unloading it)
            // instead of poisoning the mutex.
            let mut engine = match engine_guard.take() {
                Some(e) => e,
                None => {
                    return Err(anyhow::anyhow!(
                        "Model failed to load after auto-load attempt. Please check your model settings."
                    ));
                }
            };

            // Release the lock before transcribing — no mutex held during the engine call
            drop(engine_guard);

            let transcribe_result = catch_unwind(AssertUnwindSafe(
                || -> Result<transcribe_rs::TranscriptionResult> {
                    match &mut engine {
                        LoadedEngine::Whisper(whisper_engine) => {
                            let whisper_language = if validated_language == "auto" {
                                None
                            } else {
                                let normalized = if validated_language == "zh-Hans"
                                    || validated_language == "zh-Hant"
                                {
                                    "zh".to_string()
                                } else {
                                    validated_language.clone()
                                };
                                Some(normalized)
                            };

                            let params = WhisperInferenceParams {
                                language: whisper_language,
                                translate: settings.translate_to_english,
                                initial_prompt: if settings.custom_words.is_empty() {
                                    None
                                } else {
                                    Some(settings.custom_words.join(", "))
                                },
                                ..Default::default()
                            };

                            whisper_engine
                                .transcribe_with(&audio, &params)
                                .map_err(|e| anyhow::anyhow!("Whisper transcription failed: {}", e))
                        }
                        LoadedEngine::Parakeet(parakeet_engine) => {
                            let params = ParakeetParams {
                                timestamp_granularity: Some(TimestampGranularity::Segment),
                                ..Default::default()
                            };
                            parakeet_engine
                                .transcribe_with(&audio, &params)
                                .map_err(|e| {
                                    anyhow::anyhow!("Parakeet transcription failed: {}", e)
                                })
                        }
                        LoadedEngine::Moonshine(moonshine_engine) => moonshine_engine
                            .transcribe(&audio, &TranscribeOptions::default())
                            .map_err(|e| anyhow::anyhow!("Moonshine transcription failed: {}", e)),
                        LoadedEngine::MoonshineStreaming(streaming_engine) => streaming_engine
                            .transcribe(&audio, &TranscribeOptions::default())
                            .map_err(|e| {
                                anyhow::anyhow!("Moonshine streaming transcription failed: {}", e)
                            }),
                        LoadedEngine::SenseVoice(sense_voice_engine) => {
                            let language = match validated_language.as_str() {
                                "zh" | "zh-Hans" | "zh-Hant" => Some("zh".to_string()),
                                "en" => Some("en".to_string()),
                                "ja" => Some("ja".to_string()),
                                "ko" => Some("ko".to_string()),
                                "yue" => Some("yue".to_string()),
                                _ => None,
                            };
                            let params = SenseVoiceParams {
                                language,
                                use_itn: Some(true),
                            };
                            sense_voice_engine
                                .transcribe_with(&audio, &params)
                                .map_err(|e| {
                                    anyhow::anyhow!("SenseVoice transcription failed: {}", e)
                                })
                        }
                        LoadedEngine::GigaAM(gigaam_engine) => gigaam_engine
                            .transcribe(&audio, &TranscribeOptions::default())
                            .map_err(|e| anyhow::anyhow!("GigaAM transcription failed: {}", e)),
                        LoadedEngine::Canary(canary_engine) => {
                            let lang = if validated_language == "auto" {
                                None
                            } else {
                                Some(validated_language.clone())
                            };
                            let options = TranscribeOptions {
                                language: lang,
                                translate: settings.translate_to_english,
                            };
                            canary_engine
                                .transcribe(&audio, &options)
                                .map_err(|e| anyhow::anyhow!("Canary transcription failed: {}", e))
                        }
                        LoadedEngine::GroqWhisper => {
                            let api_key = settings
                                .post_process_api_keys
                                .get("groq")
                                .cloned()
                                .unwrap_or_default();
                            if api_key.is_empty() {
                                return Err(anyhow::anyhow!(
                                    "Groq API key not set. Add it in post-processing settings."
                                ));
                            }
                            let language = if validated_language == "auto" {
                                None
                            } else {
                                Some(validated_language.clone())
                            };
                            // Prefer the pre-saved WAV file (proven quality) over rebuilding from samples
                            let result = if let Some(ref path) = wav_hint {
                                transcribe_via_groq_file(path, &api_key, language.as_deref())
                            } else {
                                transcribe_via_groq(&audio, &api_key, language.as_deref())
                            };
                            result
                                .map(|text| transcribe_rs::TranscriptionResult {
                                    text,
                                    segments: None,
                                })
                                .map_err(|e| anyhow::anyhow!("Groq transcription failed: {}", e))
                        }
                    }
                },
            ));

            match transcribe_result {
                Ok(inner_result) => {
                    // Success or normal error — put the engine back
                    let mut engine_guard = self.lock_engine();
                    *engine_guard = Some(engine);
                    inner_result?
                }
                Err(panic_payload) => {
                    // Engine panicked — do NOT put it back (it's in an unknown state).
                    // The engine is dropped here, effectively unloading it.
                    let panic_msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                        s.to_string()
                    } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "unknown panic".to_string()
                    };
                    error!(
                        "Transcription engine panicked: {}. Model has been unloaded.",
                        panic_msg
                    );

                    // Clear the model ID so it will be reloaded on next attempt
                    {
                        let mut current_model = self
                            .current_model_id
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        *current_model = None;
                    }

                    let _ = self.app_handle.emit(
                        "model-state-changed",
                        ModelStateEvent {
                            event_type: "unloaded".to_string(),
                            model_id: None,
                            model_name: None,
                            error: Some(format!("Engine panicked: {}", panic_msg)),
                        },
                    );

                    return Err(anyhow::anyhow!(
                        "Transcription engine panicked: {}. The model has been unloaded and will reload on next attempt.",
                        panic_msg
                    ));
                }
            }
        };

        // Apply word correction if custom words are configured.
        // Skip for Whisper models since custom words are already passed as initial_prompt.
        let is_whisper = self
            .model_manager
            .get_model_info(&settings.selected_model)
            .map(|info| matches!(info.engine_type, EngineType::Whisper))
            .unwrap_or(false);

        let corrected_result = if !settings.custom_words.is_empty() && !is_whisper {
            apply_custom_words(
                &result.text,
                &settings.custom_words,
                settings.word_correction_threshold,
            )
        } else {
            result.text
        };

        // Filter out filler words and hallucinations
        let filtered_result = filter_transcription_output(
            &corrected_result,
            &settings.app_language,
            &settings.custom_filler_words,
        );

        let et = std::time::Instant::now();
        let translation_note = if settings.translate_to_english {
            " (translated)"
        } else {
            ""
        };
        info!(
            "Transcription completed in {}ms{}",
            (et - st).as_millis(),
            translation_note
        );

        // For Groq Whisper, add paragraph breaks after all filtering is done
        let final_result = if matches!(
            self.lock_engine().as_ref(),
            Some(LoadedEngine::GroqWhisper)
        ) {
            add_paragraphs(&filtered_result, "")
        } else {
            filtered_result
        };

        if final_result.is_empty() {
            info!("Transcription result is empty");
        } else {
            info!("Transcription result: {}", final_result);
        }

        self.maybe_unload_immediately("transcription");

        Ok(final_result)
    }

    /// Transcribe a pre-saved WAV file via Groq Whisper API.
    /// Applies the same filtering and paragraph logic as `transcribe()`.
    /// Returns Err if the model is not GroqWhisper or the API key is missing.
    pub fn transcribe_groq_wav(&self, wav_path: &std::path::Path) -> Result<String> {
        let settings = get_settings(&self.app_handle);
        let api_key = settings
            .post_process_api_keys
            .get("groq")
            .cloned()
            .unwrap_or_default();
        if api_key.is_empty() {
            return Err(anyhow::anyhow!("Groq API key not set"));
        }
        let language = if settings.selected_language == "auto" {
            None
        } else {
            Some(settings.selected_language.clone())
        };

        // Resolve VAD model path so we can trim trailing silence before sending audio to Whisper.
        // Whisper hallucinates ("Субтитры…", "Продолжение следует", etc.) when it sees silence —
        // VAD-trim eliminates the trigger at the audio level instead of post-filtering text.
        let vad_path = self
            .app_handle
            .path()
            .resolve(
                "resources/models/silero_vad_v4.onnx",
                tauri::path::BaseDirectory::Resource,
            )
            .ok();

        let raw = transcribe_groq_wav_in_chunks(wav_path, &api_key, language.as_deref(), vad_path.as_deref())?;

        let cleaned = if settings.post_process_enabled {
            llm_post_process(&raw, &api_key).unwrap_or_else(|e| {
                error!("LLM post-process failed, using raw: {}", e);
                raw.clone()
            })
        } else {
            raw.clone()
        };

        let filtered = filter_transcription_output(&cleaned, &settings.app_language, &settings.custom_filler_words);
        let final_result = add_paragraphs(&filtered, "");

        if final_result.is_empty() {
            info!("Groq WAV transcription result is empty");
        } else {
            info!("Groq WAV transcription result: {}", final_result);
        }
        Ok(final_result)
    }
}

/// Apply the user's accelerator preferences to the transcribe-rs global atomics.
/// Called on startup and whenever the user changes the setting.
pub fn apply_accelerator_settings(app: &tauri::AppHandle) {
    use transcribe_rs::accel;

    let settings = get_settings(app);

    let whisper_pref = match settings.whisper_accelerator {
        WhisperAcceleratorSetting::Auto => accel::WhisperAccelerator::Auto,
        WhisperAcceleratorSetting::Cpu => accel::WhisperAccelerator::CpuOnly,
        WhisperAcceleratorSetting::Gpu => accel::WhisperAccelerator::Gpu,
    };
    accel::set_whisper_accelerator(whisper_pref);
    accel::set_whisper_gpu_device(settings.whisper_gpu_device);
    info!(
        "Whisper accelerator set to: {}, gpu_device: {}",
        whisper_pref,
        if settings.whisper_gpu_device == accel::GPU_DEVICE_AUTO {
            "auto".to_string()
        } else {
            settings.whisper_gpu_device.to_string()
        }
    );

    let ort_pref = match settings.ort_accelerator {
        OrtAcceleratorSetting::Auto => accel::OrtAccelerator::Auto,
        OrtAcceleratorSetting::Cpu => accel::OrtAccelerator::CpuOnly,
        OrtAcceleratorSetting::Cuda => accel::OrtAccelerator::Cuda,
        OrtAcceleratorSetting::DirectMl => accel::OrtAccelerator::DirectMl,
        OrtAcceleratorSetting::Rocm => accel::OrtAccelerator::Rocm,
    };
    accel::set_ort_accelerator(ort_pref);
    info!("ORT accelerator set to: {}", ort_pref);
}

#[derive(Serialize, Clone, Debug, Type)]
pub struct GpuDeviceOption {
    pub id: i32,
    pub name: String,
    pub total_vram_mb: usize,
}

static GPU_DEVICES: OnceLock<Vec<GpuDeviceOption>> = OnceLock::new();

fn cached_gpu_devices() -> &'static [GpuDeviceOption] {
    use transcribe_rs::whisper_cpp::gpu::list_gpu_devices;

    GPU_DEVICES.get_or_init(|| {
        list_gpu_devices()
            .into_iter()
            .map(|d| GpuDeviceOption {
                id: d.id,
                name: d.name,
                total_vram_mb: d.total_vram / (1024 * 1024),
            })
            .collect()
    })
}

#[derive(Serialize, Clone, Debug, Type)]
pub struct AvailableAccelerators {
    pub whisper: Vec<String>,
    pub ort: Vec<String>,
    pub gpu_devices: Vec<GpuDeviceOption>,
}

/// Return which accelerators are compiled into this build.
pub fn get_available_accelerators() -> AvailableAccelerators {
    use transcribe_rs::accel::OrtAccelerator;

    let ort_options: Vec<String> = OrtAccelerator::available()
        .into_iter()
        .map(|a| a.to_string())
        .collect();

    let whisper_options = vec!["auto".to_string(), "cpu".to_string(), "gpu".to_string()];

    AvailableAccelerators {
        whisper: whisper_options,
        ort: ort_options,
        gpu_devices: cached_gpu_devices().to_vec(),
    }
}

impl Drop for TranscriptionManager {
    fn drop(&mut self) {
        // Skip shutdown unless this is the very last clone. TranscriptionManager
        // is cloned by initiate_model_load() and the watcher thread — those
        // clones dropping must not kill the watcher. The watcher thread holds
        // its own clone, so engine's strong_count is always >= 2 while the
        // watcher is alive. When it reaches 1, only this instance remains
        // and we can safely shut down.
        if Arc::strong_count(&self.engine) > 1 {
            return;
        }

        // Signal the watcher thread to shutdown
        self.shutdown_signal.store(true, Ordering::Relaxed);

        // Wait for the thread to finish gracefully
        if let Some(handle) = self.watcher_handle.lock().unwrap().take() {
            if let Err(e) = handle.join() {
                warn!("Failed to join idle watcher thread: {:?}", e);
            } else {
                debug!("Idle watcher thread joined successfully");
            }
        }
    }
}

/// Convert f32 audio samples to WAV bytes (16-bit PCM, mono).
fn audio_to_wav(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    use hound::{SampleFormat, WavSpec, WavWriter};
    use std::io::Cursor;

    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer = WavWriter::new(&mut cursor, spec)
            .map_err(|e| anyhow::anyhow!("WAV writer error: {}", e))?;
        for &sample in samples {
            let s = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
            writer
                .write_sample(s)
                .map_err(|e| anyhow::anyhow!("WAV write error: {}", e))?;
        }
        writer
            .finalize()
            .map_err(|e| anyhow::anyhow!("WAV finalize error: {}", e))?;
    }
    Ok(cursor.into_inner())
}

/// Call Groq Whisper API using an already-saved WAV file.
/// This avoids rebuilding the WAV from samples, which can produce different results.
fn transcribe_via_groq_file(
    wav_path: &std::path::Path,
    api_key: &str,
    language: Option<&str>,
) -> Result<String> {
    use std::process::Command;

    let mut args: Vec<String> = vec![
        "--noproxy".into(), "*".into(),
        "-s".into(), "--fail-with-body".into(),
        "-X".into(), "POST".into(),
        "https://api.groq.com/openai/v1/audio/transcriptions".into(),
        "-H".into(), format!("Authorization: Bearer {}", api_key),
        "-F".into(), "model=whisper-large-v3".into(),
        "-F".into(), "response_format=text".into(),
        "-F".into(), format!(
            "file=@{};filename=audio.wav;type=audio/wav",
            wav_path.to_str().unwrap_or("/tmp/handy_groq_audio.wav")
        ),
        "-F".into(), "temperature=0".into(),
    ];

    if let Some(lang) = language {
        args.push("-F".into());
        args.push(format!("language={}", lang));
    }

    let max_retries = 3;
    for attempt in 0..max_retries {
        let output = Command::new("curl")
            .args(&args)
            .output()
            .map_err(|e| anyhow::anyhow!("curl error: {}", e))?;

        if output.status.success() {
            let raw = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !raw.is_empty() {
                return Ok(apply_word_fixes(&raw));
            }
        }

        if attempt < max_retries - 1 {
            std::thread::sleep(std::time::Duration::from_secs(1));
            continue;
        }

        let body = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow::anyhow!("Groq API error (after {} retries): {} {}", max_retries, body, stderr));
    }
    unreachable!()
}

/// Trim trailing silence from i16 samples using Silero VAD.
/// Returns (trimmed_samples, trimmed_secs) — trimmed_secs is how much was cut off.
/// Whisper hallucinates on silence: "Субтитры создавал…", "Продолжение следует…", "Спасибо за просмотр".
/// Removing silence at audio level eliminates the root trigger.
fn trim_trailing_silence_i16(
    samples: Vec<i16>,
    sample_rate: u32,
    channels: u16,
    vad_path: &std::path::Path,
) -> (Vec<i16>, f64) {
    use crate::audio_toolkit::vad::{SileroVad, VoiceActivityDetector};

    // Silero is trained on 16kHz mono. If our WAV is something else, skip trim.
    if sample_rate != 16000 || channels != 1 {
        return (samples, 0.0);
    }

    let mut vad = match SileroVad::new(vad_path, 0.3) {
        Ok(v) => v,
        Err(e) => {
            warn!("VAD init failed, skipping silence trim: {}", e);
            return (samples, 0.0);
        }
    };

    const FRAME: usize = 480; // 30ms @ 16kHz
    // Scan all frames, record index of last speech frame
    let mut last_speech_frame: Option<usize> = None;
    for (i, frame) in samples.chunks(FRAME).enumerate() {
        if frame.len() != FRAME {
            break;
        }
        let frame_f32: Vec<f32> = frame.iter().map(|&s| s as f32 / 32768.0).collect();
        if let Ok(fr) = vad.push_frame(&frame_f32) {
            if fr.is_speech() {
                last_speech_frame = Some(i);
            }
        }
    }

    let Some(last) = last_speech_frame else {
        // No speech detected at all — return as-is to avoid producing empty audio
        return (samples, 0.0);
    };

    // Keep speech + 300ms tail margin (= 10 frames)
    let keep_frames = (last + 10).min(samples.len() / FRAME);
    let keep_samples = keep_frames * FRAME;
    if keep_samples >= samples.len() {
        return (samples, 0.0);
    }
    let trimmed_secs = (samples.len() - keep_samples) as f64 / sample_rate as f64;
    let mut trimmed = samples;
    trimmed.truncate(keep_samples);
    (trimmed, trimmed_secs)
}

/// Split WAV into 25-second segments using hound (no external tools required),
/// transcribe each via Groq Whisper, and join results.
/// Prevents Whisper from dropping speech at the end of long recordings.
fn transcribe_groq_wav_in_chunks(
    wav_path: &std::path::Path,
    api_key: &str,
    language: Option<&str>,
    vad_path: Option<&std::path::Path>,
) -> Result<String> {
    use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
    use std::io::Cursor;

    let mut reader = WavReader::open(wav_path)
        .map_err(|e| anyhow::anyhow!("Failed to open WAV for chunking: {}", e))?;
    let spec = reader.spec();

    // Normalise to i16 samples regardless of source format
    let raw_samples: Vec<i16> = match spec.sample_format {
        SampleFormat::Int => reader
            .samples::<i16>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| anyhow::anyhow!("WAV read error: {}", e))?,
        SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| anyhow::anyhow!("WAV read error: {}", e))?
            .into_iter()
            .map(|s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
            .collect(),
    };

    // Trim trailing silence — eliminates Whisper hallucination triggers
    let all_samples = if let Some(vp) = vad_path {
        let (trimmed, cut_secs) = trim_trailing_silence_i16(raw_samples, spec.sample_rate, spec.channels, vp);
        if cut_secs > 0.05 {
            info!("VAD trimmed {:.2}s of trailing silence", cut_secs);
        }
        trimmed
    } else {
        raw_samples
    };

    if all_samples.is_empty() {
        return Err(anyhow::anyhow!("No samples left after VAD trim"));
    }

    // Chunking strategy: 25s chunks, but never leave a tiny tail.
    // A trailing chunk under 5s is fed only ~1-2 seconds of speech to Whisper,
    // which causes it to fall back to memorised endings ("Продолжение следует…").
    // To avoid this, we merge a short tail into the preceding chunk — the result
    // (up to ~30s) still fits Whisper's internal window.
    let chunk_samples = (25 * spec.sample_rate * spec.channels as u32) as usize;
    let min_last_samples = (5 * spec.sample_rate * spec.channels as u32) as usize;

    let mut chunk_ranges: Vec<(usize, usize)> = Vec::new();
    let mut pos = 0;
    while pos < all_samples.len() {
        let end = (pos + chunk_samples).min(all_samples.len());
        chunk_ranges.push((pos, end));
        pos = end;
    }
    if chunk_ranges.len() >= 2 {
        let (last_start, last_end) = *chunk_ranges.last().unwrap();
        if last_end - last_start < min_last_samples {
            chunk_ranges.pop();
            chunk_ranges.last_mut().unwrap().1 = last_end;
        }
    }

    let total_chunks = chunk_ranges.len();
    info!("Splitting into {} chunks", total_chunks);

    let ts = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();

    // Returns true when a chunk result looks like Whisper hallucination rather than real speech.
    // Checks: known hallucination markers, "и так далее" as the ENTIRE result, too sparse.
    let chunk_is_bad = |text: &str, chunk_len: usize| -> bool {
        let lower = text.to_lowercase();
        if lower.contains("субтитр") || lower.contains("dimatorzok") {
            return true;
        }
        // "и так далее" as the whole chunk (not as a natural sentence ending) is a hallucination
        let trimmed_lower = text.trim().to_lowercase();
        let only_etc = trimmed_lower == "и так далее."
            || trimmed_lower == "и так далее"
            || trimmed_lower == "и т.д."
            || trimmed_lower == "и т. д.";
        if only_etc {
            return true;
        }
        // Too sparse: real speech produces ~10+ chars/sec; < 4 chars/sec on a chunk > 5s = truncated
        let chunk_secs = chunk_len as f64 / (spec.sample_rate as f64 * spec.channels as f64);
        if chunk_secs > 5.0 && (text.chars().count() as f64 / chunk_secs) < 4.0 {
            return true;
        }
        false
    };

    let mut parts: Vec<String> = Vec::new();
    for (i, &(start, end)) in chunk_ranges.iter().enumerate() {
        let chunk = &all_samples[start..end];
        // Write chunk to a temp WAV file
        let chunk_path = std::env::temp_dir().join(format!("handy_chunk_{}_{}.wav", ts, i));
        let chunk_spec = WavSpec {
            channels: spec.channels,
            sample_rate: spec.sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        {
            let mut cursor = Cursor::new(Vec::new());
            {
                let mut writer = WavWriter::new(&mut cursor, chunk_spec)
                    .map_err(|e| anyhow::anyhow!("WavWriter error: {}", e))?;
                for &s in chunk {
                    writer.write_sample(s)
                        .map_err(|e| anyhow::anyhow!("WAV write error: {}", e))?;
                }
                writer.finalize()
                    .map_err(|e| anyhow::anyhow!("WAV finalize error: {}", e))?;
            }
            std::fs::write(&chunk_path, cursor.into_inner())
                .map_err(|e| anyhow::anyhow!("Failed to write chunk: {}", e))?;
        }

        match transcribe_via_groq_file(&chunk_path, api_key, language) {
            Ok(text1) => {
                let trimmed1 = text1.trim().to_string();
                let final_text = if chunk_is_bad(&trimmed1, chunk.len()) {
                    warn!("Chunk {}/{} looks like hallucination ({}), retrying", i + 1, total_chunks, trimmed1.chars().count());
                    match transcribe_via_groq_file(&chunk_path, api_key, language) {
                        Ok(text2) => {
                            let trimmed2 = text2.trim().to_string();
                            if chunk_is_bad(&trimmed2, chunk.len()) {
                                // Both bad — take the longer one
                                if trimmed2.chars().count() > trimmed1.chars().count() { trimmed2 } else { trimmed1 }
                            } else {
                                trimmed2
                            }
                        }
                        Err(e) => { warn!("Chunk {}/{} retry failed: {}", i + 1, total_chunks, e); trimmed1 }
                    }
                } else {
                    trimmed1
                };
                if !final_text.is_empty() {
                    info!("Chunk {}/{}: {} chars", i + 1, total_chunks, final_text.chars().count());
                    parts.push(final_text);
                }
            }
            Err(e) => warn!("Chunk {}/{} transcription failed: {}", i + 1, total_chunks, e),
        }
        let _ = std::fs::remove_file(&chunk_path);
    }

    if parts.is_empty() {
        return Err(anyhow::anyhow!("All chunks failed to transcribe"));
    }

    Ok(parts.join(" "))
}

/// Call Groq Whisper API and return transcribed text.
/// Uses curl with --noproxy to bypass system proxy settings.
fn transcribe_via_groq(
    audio: &[f32],
    api_key: &str,
    language: Option<&str>,
) -> Result<String> {
    use std::io::Write;
    use std::process::Command;

    // Pad with 50ms of silence at start so Whisper doesn't clip the first word
    let silence_samples = (16000 * 50 / 1000) as usize; // 800 samples at 16kHz
    let mut padded = vec![0.0f32; silence_samples];
    padded.extend_from_slice(audio);
    let wav_data = audio_to_wav(&padded, 16000)?;

    // Write WAV to a temp file so curl can upload it
    let temp_path = std::env::temp_dir().join("handy_groq_audio.wav");
    {
        let mut f = std::fs::File::create(&temp_path)
            .map_err(|e| anyhow::anyhow!("Failed to create temp WAV: {}", e))?;
        f.write_all(&wav_data)
            .map_err(|e| anyhow::anyhow!("Failed to write temp WAV: {}", e))?;
    }

    let mut args: Vec<String> = vec![
        "--noproxy".into(), "*".into(),
        "-s".into(), "--fail-with-body".into(),
        "-X".into(), "POST".into(),
        "https://api.groq.com/openai/v1/audio/transcriptions".into(),
        "-H".into(), format!("Authorization: Bearer {}", api_key),
        "-F".into(), "model=whisper-large-v3".into(),
        "-F".into(), "response_format=text".into(),
        "-F".into(), format!(
            "file=@{};filename=audio.wav;type=audio/wav",
            temp_path.to_str().unwrap_or("/tmp/handy_groq_audio.wav")
        ),
        "-F".into(), "temperature=0".into(),
    ];

    if let Some(lang) = language {
        args.push("-F".into());
        args.push(format!("language={}", lang));
    }

    let max_retries = 3;
    let mut last_output = None;
    for attempt in 0..max_retries {
        let output = Command::new("curl")
            .args(&args)
            .output()
            .map_err(|e| anyhow::anyhow!("curl error: {}", e))?;

        if output.status.success() {
            let raw = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !raw.is_empty() {
                let _ = std::fs::remove_file(&temp_path);
                return Ok(apply_word_fixes(&raw));
            }
        }

        last_output = Some(output);
        if attempt < max_retries - 1 {
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    }

    let _ = std::fs::remove_file(&temp_path);
    let output = last_output.unwrap();
    if !output.status.success() {
        let body = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow::anyhow!("Groq API error (after {} retries): {} {}", max_retries, body, stderr));
    }

    let raw = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(apply_word_fixes(&raw))
}

/// Fix common transcription errors for specific terms.
fn apply_word_fixes(text: &str) -> String {
    use regex::Regex;

    // Each entry: (pattern, replacement) — case-insensitive match
    let fixes: &[(&str, &str)] = &[
        // Getcourse variants
        (r"(?i)\bget[\s-]?cours[a-zа-яё]*\b", "Getcourse"),
        (r"(?i)\bgit[\s-]?cours[a-zа-яё]*\b", "Getcourse"),
        (r"(?i)\bгеткурс[а-яё]*\b", "Геткурс"),
        (r"(?i)\bгит[\s-]?курс[а-яё]*\b", "Геткурс"),
        // Вагэ variants
        (r"(?i)\bваге\b", "Вагэ"),
        (r"(?i)\bвлаге\b", "Вагэ"),
        (r"(?i)\bвг\b", "Вагэ"),
        // VS Code variants
        (r"(?i)\bвисекот\b", "VS Code"),
        (r"(?i)\bвиси[\s-]?код\b", "VS Code"),
        (r"(?i)\bvisi[\s-]?code\b", "VS Code"),
        // Claude Code variants
        (r"(?i)\bcloth[\s-]?code\b", "Claude Code"),
        (r"(?i)\bcloud[\s-]?cloud\b", "Claude Code"),
        (r"(?i)\bclaw[\s-]?code\b", "Claude Code"),
        // OpenClaw
        (r"(?i)\bopen[\s-]?close\b", "OpenClaw"),
        // Lovable
        (r"(?i)\blava[\s-]?bull\b", "Lovable"),
        // SuperWhisper
        (r"(?i)\bsuper[\s-]?whisper\b", "SuperWhisper"),
        (r"(?i)\bvisper\b", "SuperWhisper"),
        // API
        (r"(?i)\bА[рp]\b", "API"),
        // md
        (r"(?i)\bмд\b", "md"),
        // референсы
        (r"(?i)референци[яи]", "референсы"),
        // скилами
        (r"(?i)скелами", "скилами"),
        // Cowork
        (r"(?i)\bclawrk\b", "Cowork"),
        (r"(?i)\bко[\s-]?ворк\b", "Cowork"),
    ];

    let mut result = text.to_string();
    for (pattern, replacement) in fixes {
        if let Ok(re) = Regex::new(pattern) {
            result = re.replace_all(&result, *replacement).to_string();
        }
    }

    // Remove trailing syllable artifacts: Whisper sometimes splits the last word across
    // chunks, leaving a short fragment glued after the final punctuation mark.
    // e.g. "все. Да.ть" → "все. Да."  (strips the dangling "ть")
    if let Ok(re) = Regex::new(r"([.!?])[а-яёА-ЯЁa-zA-Z]{1,4}$") {
        result = re.replace(&result, "$1").to_string();
    }

    result
}

/// LLM post-processing: fix punctuation, spelling (Claude, VPN, Геткурс), remove filler words.
fn llm_post_process(text: &str, api_key: &str) -> Result<String> {
    use std::process::Command;

    if text.trim().is_empty() {
        return Ok(text.to_string());
    }

    let prompt = r#"Исправь транскрипцию русской речи:
1. Исправь очевидные ошибки распознавания (например: «Clow» → «Claude», «клоу» → «Claw», «клод клоу» → «ClaudeClaw», «PPN» или «ВПН» → «VPN», «Клод Кот» или «КloдCot» → «Claude Code», «геткурс» или «ГитКур» → «Геткурс», «Getcourse»)
2. Расставь точки, запятые, вопросительные знаки
3. Расставь заглавные буквы после точек
4. Сохрани смысл и порядок слов точно
5. Верни только исправленный текст, без пояснений"#;

    let body = format!(
        r#"{{"model":"llama-3.1-8b-instant","messages":[{{"role":"system","content":"Ты корректор транскрипций. Исправляй только ошибки распознавания и пунктуацию. Не меняй смысл, не добавляй от себя."}},{{"role":"user","content":"{}\n\nТранскрипция:\n{}"}}],"temperature":0}}"#,
        prompt.replace('"', r#"\""#).replace('\n', r#"\n"#),
        text.replace('"', r#"\""#).replace('\n', r#"\n"#)
    );

    let output = Command::new("curl")
        .args(&[
            "--noproxy", "*",
            "-s", "--fail-with-body",
            "-X", "POST",
            "https://api.groq.com/openai/v1/chat/completions",
            "-H", &format!("Authorization: Bearer {}", api_key),
            "-H", "Content-Type: application/json",
            "-d", &body,
        ])
        .output()
        .map_err(|e| anyhow::anyhow!("LLM curl error: {}", e))?;

    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stdout);
        return Err(anyhow::anyhow!("LLM API error: {}", err));
    }

    let response = String::from_utf8_lossy(&output.stdout);
    // Extract content from JSON: {"choices":[{"message":{"role":"assistant","content":"..."}}]}
    // Use rfind to get the LAST "content":" which is the assistant's response, not system echo
    if let Some(start) = response.rfind(r#""content":""#) {
        let content_start = start + r#""content":""#.len();
        if let Some(end) = response[content_start..].find(r#"","refusal"#)
            .or_else(|| response[content_start..].find(r#""},"logprobs"#))
            .or_else(|| response[content_start..].find(r#""}"#))
        {
            let content = &response[content_start..content_start + end];
            let unescaped = content
                .replace(r#"\n"#, "\n")
                .replace(r#"\""#, "\"")
                .replace(r#"\\"#, "\\");
            return Ok(unescaped);
        }
    }

    // Fallback: return raw text if parsing fails
    Err(anyhow::anyhow!("Failed to parse LLM response"))
}

/// Break long transcription into paragraphs: every 3 sentences = new paragraph.
fn add_paragraphs(text: &str, _api_key: &str) -> String {
    if text.chars().count() < 200 {
        return text.to_string();
    }

    // Split into sentences on . ! ? — but only flush at the LAST terminator in a run
    // (so "?!" or "..." counts as ONE end-of-sentence, not 2-3).
    let chars: Vec<char> = text.chars().collect();
    let is_term = |c: char| matches!(c, '.' | '!' | '?');
    let mut sentences: Vec<String> = Vec::new();
    let mut current = String::new();
    for (i, &ch) in chars.iter().enumerate() {
        current.push(ch);
        // Flush only if this is a terminator AND the next char isn't also a terminator.
        if is_term(ch) && chars.get(i + 1).map_or(true, |&nc| !is_term(nc)) {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }
    if !current.trim().is_empty() {
        sentences.push(current.trim().to_string());
    }

    // If no punctuation found (1 giant "sentence"), split by commas instead
    if sentences.len() <= 1 && text.chars().count() > 300 {
        let parts: Vec<&str> = text.split(',').collect();
        if parts.len() > 1 {
            let mut paragraphs: Vec<String> = Vec::new();
            let mut current_para = String::new();
            for part in &parts {
                if !current_para.is_empty() {
                    current_para.push(',');
                }
                current_para.push_str(part);
                if current_para.chars().count() > 300 {
                    paragraphs.push(current_para.trim().to_string());
                    current_para = String::new();
                }
            }
            if !current_para.trim().is_empty() {
                paragraphs.push(current_para.trim().to_string());
            }
            return paragraphs.join("\n\n");
        }
    }

    // Group every 3 sentences into a paragraph
    sentences
        .chunks(3)
        .map(|chunk| chunk.join(" "))
        .collect::<Vec<_>>()
        .join("\n\n")
}
