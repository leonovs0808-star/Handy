# Наши правки к Handy

Форк [cjpais/Handy](https://github.com/cjpais/Handy) с кастомными правками для голосового ввода на русском языке через Groq Whisper.

## Origin

- **Upstream:** `cjpais/Handy` на GitHub
- **База:** тег `v0.8.1`
- **Ветка с нашими правками:** `our-patches`
- **Наш fork:** `leonovs0808-star/Handy`

## Что добавили поверх upstream

Upstream Handy **не поддерживает Groq Whisper** как бэкенд транскрипции — у них Groq только для LLM-постпроцессинга текста. Мы добавили полноценный Groq Whisper бэкенд для русской речи.

### Новая модель `groq-whisper`

Зарегистрирована в `src-tauri/src/managers/model.rs` как отдельная модель. Выбирается в UI в разделе Models.

### Как работает Groq Whisper ветка

1. `actions.rs::stop()` — проверяет `selected_model == "groq-whisper"`. Если да:
   - Сохраняет WAV-файл на диск (`audio_toolkit::save_wav_file`)
   - Вызывает `transcription_manager::transcribe_groq_wav(&wav_path)`
   - Фолбэк на обычный `transcribe(samples)` если сохранение WAV не удалось

2. `managers/transcription.rs`:
   - `transcribe_groq_wav(wav_path)` — публичный метод: берёт API ключ из настроек, вызывает `transcribe_via_groq_file`, прогоняет через фильтры, добавляет абзацы
   - `transcribe_via_groq_file(wav_path, api_key, language)` — приватная функция: отправляет сохранённый WAV в Groq через `curl` multipart
   - `transcribe_via_groq(audio, ...)` — старая функция (fallback): строит WAV в памяти и отправляет
   - `add_paragraphs(text, _)` — разбивка на абзацы по длине для длинных транскрипций

### Настройки Groq API запроса

```
model=whisper-large-v3
response_format=text
temperature=0
language=<из настроек>
file=@<saved_wav>
```

**Важно:** `prompt` параметр **НЕ используется**. Раньше пробовали добавлять подсказки типа "Геткурс, Getcourse, Claude, VS Code, VPN", но Whisper начинал вставлять эти слова прямо в текст как галлюцинации при неуверенности. Без промпта на русской речи работает чище.

### Multipart upload

- В `Cargo.toml` добавлена feature `multipart` для `reqwest`
- Запрос отправляется через системный `curl` с `--noproxy '*'` (избегаем системных прокси)

### FoundationModels stub

В `build.rs` принудительно отключена сборка с Apple FoundationModels (`has_foundation_models = ... && false`). Плагин FoundationModelsMacros недоступен в текущей SDK на машине разработки.

## Сборка и установка

### Требования

- Bun (для сборки фронтенда)
- Rust + Cargo
- CMake с обходом политики версии

### Сборка

```bash
cd Handy-fork
CMAKE_POLICY_VERSION_MINIMUM=3.5 bun run tauri build --no-bundle
```

Сборка занимает 5-9 минут. Результат: `src-tauri/target/release/handy`.

### Установка в /Applications/Handy.app

```bash
# 1. Заменить бинарник
cp src-tauri/target/release/handy /Applications/Handy.app/Contents/MacOS/handy

# 2. Переподписать (ad-hoc)
codesign --force --deep --sign - \
  --entitlements src-tauri/Entitlements.plist \
  /Applications/Handy.app

# 3. Перезапустить приложение
pkill -x handy
sleep 1
open /Applications/Handy.app
```

## Настройки Handy для оптимальной работы

После установки в UI приложения:

1. **Models** → выбрать `Groq Whisper` (наша модель)
2. **Общие → Язык** → `Russian`
3. **Продвинутые → Метод вставки** → `Буфер обмена (Cmd+V)` ⚠️
4. **Продвинутые → Обработка буфера обмена** → НЕ `Не изменять буфер`

### Почему paste_method критичен

`Direct` метод печатает посимвольно через эмуляцию клавиатуры — на длинных текстах теряет/дублирует символы, получается мусор типа "неделаааомент". Надо `Cmd+V` (буфер обмена).

### Почему обработка буфера не `Не изменять`

Если `paste_method=ctrl_v` и одновременно `clipboard_handling=dont_modify` — возникает race condition, вставляется старое содержимое буфера, а не свежая транскрипция.

## Известные ограничения Whisper

- **Галлюцинации с числами** на длинных записях (5+ мин) — иногда вставляет "100 50 5000..." вместо реальной речи. Чинится только LLM-постпроцессингом (включается в `Экспериментальные функции`).
- **Галлюцинация "Субтитры создавал DimaTorzok"** в конце — Whisper так "закрывает" аудио. Фильтруется в `filter_transcription_output`.
- **Неправильное распознавание специфичных слов** ("кладом" вместо "Клодом") — лечится только LLM-постпроцессингом.

## Связанный проект

Этот форк — часть инфраструктуры транскрипции. Основной Telegram-бот живёт в соседнем репозитории [transcribe-bot](https://github.com/leonovs0808-star/transcribe-bot).
