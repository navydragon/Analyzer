# Инструкция по деплою на продакшене

## После пулла изменений

### 1. Проверка конфигурации systemd service

#### Найти и проверить service файл:

```bash
# Найти service файл (обычно в /etc/systemd/system/ или /lib/systemd/system/)
sudo systemctl status analyzer-api | grep "Loaded:"
# или
ls -la /etc/systemd/system/analyzer-api.service
ls -la /lib/systemd/system/analyzer-api.service

# Просмотреть содержимое service файла
sudo cat /etc/systemd/system/analyzer-api.service
# или
sudo systemctl cat analyzer-api
```

#### Проверить команду запуска:

В service файле должна быть команда запуска **БЕЗ** `--reload` и **БЕЗ** `--workers` (для кэширования моделей):

**Правильно:**
```ini
ExecStart=/path/to/venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Неправильно:**
```ini
# НЕ используйте --reload в продакшене
ExecStart=/path/to/venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# НЕ используйте --no-reload (эта опция не существует)
ExecStart=/path/to/venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --no-reload

# НЕ используйте --workers (каждый воркер имеет свой кэш, модели будут загружаться в каждом)
ExecStart=/path/to/venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Важно:**
- uvicorn не поддерживает опцию `--no-reload`. Просто не указывайте `--reload` в команде запуска.
- **НЕ используйте `--workers`** - каждый воркер имеет свой собственный процесс и свой собственный кэш. Модели будут загружаться в каждом воркере отдельно, что увеличит использование памяти и время загрузки.

#### Проверить переменные окружения:

В секции `[Service]` проверьте переменные окружения:

```bash
# Просмотреть все переменные окружения сервиса
sudo systemctl show analyzer-api | grep Environment
```

В service файле должно быть:
```ini
[Service]
Environment="UVICORN_RELOAD=false"
# или переменная вообще не должна быть установлена
```

**Если переменная `UVICORN_RELOAD=true` или `--reload` в команде:**
1. Отредактируйте service файл: `sudo nano /etc/systemd/system/analyzer-api.service`
2. Уберите `--reload` из команды `ExecStart` (просто не указывайте эту опцию)
3. Уберите `--no-reload` если он есть (эта опция не существует в uvicorn)
4. Установите `Environment="UVICORN_RELOAD=false"` или удалите эту строку
5. Перезагрузите конфигурацию: `sudo systemctl daemon-reload`
6. Перезапустите сервис: `sudo systemctl restart analyzer-api`

#### Проверить текущие переменные окружения процесса:

```bash
# Найти PID процесса
sudo systemctl status analyzer-api | grep "Main PID"

# Проверить переменные окружения запущенного процесса
sudo cat /proc/$(systemctl show analyzer-api -p MainPID --value)/environ | tr '\0' '\n' | grep UVICORN
```

#### Проверить логи при старте:

```bash
# Просмотреть логи при старте (должны быть сообщения о PID и переменных окружения)
sudo journalctl -u analyzer-api -n 50 --no-pager | grep -E "PID|UVICORN_RELOAD|Запуск приложения"
```

В логах должно быть:
```
PID процесса: <число>
UVICORN_RELOAD=false
```

#### Проверить, не используется ли несколько воркеров:

```bash
# Проверить, сколько процессов uvicorn запущено
ps aux | grep uvicorn | grep -v grep

# Проверить PID процессов при запросах
sudo journalctl -u analyzer-api --since "10 minutes ago" --no-pager | grep "PID=" | sort -u
```

**Если используется несколько воркеров:**
- Каждый воркер имеет свой собственный процесс и свой собственный кэш
- Модели будут загружаться в каждом воркере отдельно
- Это увеличит использование памяти (каждая модель занимает ~500 MB для small)
- Для кэширования моделей рекомендуется использовать **один процесс без воркеров**

### 2. Очистка кэша Python (опционально, но рекомендуется)

```bash
cd /path/to/Analyzer
find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
```

### 3. Перезапуск сервиса

#### Если используется systemd:

```bash
# Проверить статус
sudo systemctl status analyzer-api

# Перезапустить сервис
sudo systemctl restart analyzer-api

# Проверить логи
sudo journalctl -u analyzer-api -f

# Проверить логи с момента последнего запуска
sudo journalctl -u analyzer-api --since "5 minutes ago" --no-pager
```

#### Если используется supervisor:

```bash
# Перезапустить процесс
supervisorctl restart analyzer-api

# Проверить статус
supervisorctl status analyzer-api

# Проверить логи
tail -f /path/to/logs/analyzer-api.log
```

#### Если запускается напрямую через uvicorn:

```bash
# Остановить текущий процесс (Ctrl+C или kill)
# Затем запустить заново:
cd /path/to/Analyzer
source venv/bin/activate  # если используется venv
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 4. Проверка работоспособности

#### Проверить health endpoint:

```bash
curl http://localhost:8000/v1/health
```

Ожидаемый ответ:
```json
{"status": "ok", "ffmpeg": true}
```

#### Проверить версию:

```bash
curl http://localhost:8000/v1/version
```

#### Проверить логи на наличие ошибок:

```bash
# systemd
sudo journalctl -u analyzer-api -n 100 --no-pager

# Проверить логи на наличие сообщений о кэше
sudo journalctl -u analyzer-api -n 200 --no-pager | grep -E "кэш|cache|Анализатор|analyzer"

# Проверить, не перезапускается ли процесс (PID должен быть постоянным)
sudo journalctl -u analyzer-api --since "1 hour ago" --no-pager | grep "PID процесса"

# или если логи в файле
tail -n 100 /path/to/logs/app.log
```

### 5. Мониторинг первого запроса

При первом запросе на транскрипцию модель Whisper будет загружаться (может занять 10-30 секунд в зависимости от размера модели). Это нормально.

Проверьте логи:
```bash
# Должны появиться сообщения:
# "Загрузка модели Whisper: small (это может занять время при первом запуске)..."
# "Модель Whisper small загружена успешно"
```

### 6. Проверка подавления предупреждений NNPACK

В логах не должно быть сообщений:
```
[W...] Could not initialize NNPACK! Reason: Unsupported hardware.
```

Если они все еще появляются, это не критично (модель работает), но можно дополнительно настроить фильтрацию на уровне systemd.

### 7. Проверка работы транскрипции

Сделайте тестовый запрос:

```bash
# Простой тест (замените на реальный аудиофайл)
curl -X POST http://localhost:8000/v1/voice/analyze \
  -F "file=@test.wav" \
  -F "reference=тестовый текст" \
  -F "model_size=small" \
  -F "language=ru"
```

## Важные замечания

1. **Первый запрос будет медленным** - это нормально, модель загружается в память
2. **Модели кэшируются** - последующие запросы будут быстрее
3. **Stanza инициализируется лениво** - первый запрос к `/v1/text/grammar` займет 15-20 секунд
4. **Reload отключен по умолчанию** - это предотвращает перезапуск процесса во время работы

## Откат изменений (если что-то пошло не так)

```bash
# Вернуться к предыдущему коммиту
git log --oneline -10  # найти нужный коммит
git checkout <commit-hash>
sudo systemctl restart analyzer-api
```

## Дополнительная настройка (опционально)

### Установка переменных окружения в systemd service

Если используете systemd, добавьте в service файл `/etc/systemd/system/analyzer-api.service`:

```ini
[Service]
Environment="PYTORCH_ENABLE_MPS_FALLBACK=1"
Environment="UVICORN_RELOAD=false"
Environment="MODEL_SIZE=small"
Environment="LANGUAGE=ru"
# Другие переменные окружения
```

После изменения service файла:
```bash
# Перезагрузить конфигурацию systemd
sudo systemctl daemon-reload

# Перезапустить сервис
sudo systemctl restart analyzer-api

# Проверить, что изменения применились
sudo systemctl show analyzer-api | grep Environment
```

### Пример правильного systemd service файла

```ini
[Unit]
Description=Analyzer API Service
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/Analyzer
Environment="PYTORCH_ENABLE_MPS_FALLBACK=1"
Environment="UVICORN_RELOAD=false"
Environment="MODEL_SIZE=small"
Environment="LANGUAGE=ru"
ExecStart=/path/to/venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Важно:**
- В `ExecStart` НЕ должно быть `--reload`!
- НЕ используйте `--no-reload` (эта опция не существует в uvicorn)
- Просто не указывайте `--reload` в команде запуска

### Подавление предупреждений NNPACK на уровне systemd

Предупреждения NNPACK идут напрямую из C++ кода PyTorch в файловый дескриптор stderr, минуя Python-фильтры. Для их подавления можно использовать фильтрацию на уровне systemd:

#### Вариант 1: Фильтрация при просмотре логов (рекомендуется)

```bash
# Просмотр логов без предупреждений NNPACK
sudo journalctl -u analyzer-api -f | grep -v NNPACK

# Или использовать более сложный фильтр
sudo journalctl -u analyzer-api -f | grep -v -E "NNPACK|Could not initialize NNPACK"
```

#### Вариант 2: Перенаправление stderr в systemd service (осторожно!)

Если предупреждения NNPACK мешают, можно добавить в service файл:

```ini
[Service]
# Перенаправление stderr в /dev/null (подавляет все stderr, включая NNPACK)
# ВНИМАНИЕ: Это также скроет важные ошибки!
StandardError=null
```

**Внимание:** Использование `StandardError=null` подавит все сообщения stderr, включая важные ошибки. Используйте только если предупреждения NNPACK действительно мешают, и вы уверены, что важные ошибки логируются через Python logging.

#### Вариант 3: Использование Python-фильтра (уже реализовано)

В коде уже реализован глобальный фильтр stderr, который должен перехватывать большинство предупреждений. Если предупреждения все еще появляются в логах, они идут напрямую из C++ кода и требуют фильтрации на уровне systemd (вариант 1 или 2).

**Рекомендация:** Используйте вариант 1 (фильтрация при просмотре логов) - это безопасно и не скрывает важные ошибки.

### Мониторинг использования памяти

Модели Whisper занимают память:
- `tiny`: ~75 MB
- `base`: ~150 MB
- `small`: ~500 MB
- `medium`: ~1.5 GB
- `large`: ~3 GB

Убедитесь, что на сервере достаточно памяти для выбранной модели.
