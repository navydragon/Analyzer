# Инструкция по деплою на продакшене

## После пулла изменений

### 1. Проверка переменных окружения

Убедитесь, что `UVICORN_RELOAD` не установлен или установлен в `false`:

```bash
# Проверить текущее значение
echo $UVICORN_RELOAD

# Если установлен в true, отключить (для продакшена)
unset UVICORN_RELOAD
# или в systemd service файле убедиться, что переменная не установлена
```

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
sudo systemctl status your-service-name

# Перезапустить сервис
sudo systemctl restart your-service-name

# Проверить логи
sudo journalctl -u your-service-name -f
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
sudo journalctl -u your-service-name -n 100 --no-pager

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
sudo systemctl restart your-service-name
```

## Дополнительная настройка (опционально)

### Установка переменных окружения в systemd service

Если используете systemd, добавьте в service файл:

```ini
[Service]
Environment="PYTORCH_ENABLE_MPS_FALLBACK=1"
Environment="UVICORN_RELOAD=false"
# Другие переменные окружения
```

### Мониторинг использования памяти

Модели Whisper занимают память:
- `tiny`: ~75 MB
- `base`: ~150 MB
- `small`: ~500 MB
- `medium`: ~1.5 GB
- `large`: ~3 GB

Убедитесь, что на сервере достаточно памяти для выбранной модели.
