# Dictado por voz con Whisper

Herramienta de dictado multiplataforma: pulsás una tecla, hablás, y el texto que dijiste
aparece pegado donde tengas el cursor. La transcripción corre **localmente** en tu equipo
— no envía audio a ningún servidor.

## Cómo funciona

1. El programa deja el micrófono escuchando en segundo plano.
2. Pulsás **F5** → empieza a grabar (`● Grabando...`).
3. Pulsás **F5** de nuevo → para de grabar y transcribe (`■ Transcribiendo...`).
4. El texto reconocido:
   - se **imprime en la consola** para que veas la calidad de la transcripción, y
   - se **pega automáticamente** en la app que tengas activa (portapapeles + `Cmd/Ctrl+V`).
5. Si no se pudo pegar (por ejemplo, falta el permiso de Accesibilidad en macOS), te avisa
   por consola y el texto queda igual en el portapapeles listo para pegar a mano.

Solo se puede grabar/transcribir de a una por vez: si volvés a pulsar F5 mientras todavía
está transcribiendo lo anterior, te avisa `… ocupado, esperá a que termine` en lugar de
mezclar grabaciones.

Para salir: **Ctrl+C** en la terminal.

## Compatibilidad

Funciona en **macOS, Windows y Linux**. El motor de transcripción se elige automáticamente
según la plataforma:

| Plataforma            | Motor           | Modelos                                                        |
|-----------------------|-----------------|---------------------------------------------------------------|
| **macOS** (Apple Silicon) | `mlx-whisper`   | Usa los modelos MLX locales de `~/AI` sobre la GPU del Mac. No descarga nada. |
| **Windows / Linux**   | `faster-whisper`| Descarga el modelo la primera vez y lo cachea. Corre en CPU o GPU NVIDIA (CUDA). |

El pegado del texto también es multiplataforma: usa el portapapeles del sistema y simula
`Cmd+V` en macOS o `Ctrl+V` en Windows/Linux.

## Requisitos

- **Python 3.9+**.
- **macOS:** chip Apple Silicon (M1/M2/M3…) y los modelos Whisper MLX en `~/AI`.
- **Windows / Linux:** conexión a internet la primera vez (para descargar el modelo).
  Opcional: GPU NVIDIA con CUDA para acelerar; si no, corre en CPU.

## Instalación

```bash
# 1. Crear y activar un entorno virtual
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows (PowerShell)

# 2. Instalar dependencias (pip elige el motor según tu sistema operativo)
pip install -r requirements.txt

# 3. Copiar la config de ejemplo y ajustarla si hace falta
cp .env.example .env               # en Windows: copy .env.example .env
```

### Permisos de macOS (solo Mac)

Para que pueda escuchar el micrófono y simular el pegado, hay que darle permisos a la
terminal (o al editor) desde la que ejecutás el script:

- **Ajustes del Sistema → Privacidad y seguridad → Micrófono** → activá tu terminal.
- **Ajustes del Sistema → Privacidad y seguridad → Accesibilidad** → activá tu terminal.

Sin el permiso de Accesibilidad el texto se transcribe igual, pero no se pega solo
(queda en el portapapeles).

## Uso

```bash
python main.py
```

Deberías ver algo como:

```
Plataforma: macOS (MLX)
Modelo: mlx-community/whisper-large-v3-turbo
Listo. Pulsá Key.f5 para grabar/parar. Ctrl+C para salir.
```

A partir de ahí, F5 para grabar/parar.

## Configuración (`.env`)

| Variable        | Descripción                                            | Por defecto                                    |
|-----------------|--------------------------------------------------------|------------------------------------------------|
| `WHISPER_MODEL` | Modelo de Whisper a usar.                              | macOS: `mlx-community/whisper-large-v3-turbo` · Windows/Linux: `large-v3` |
| `WHISPER_LANG`  | Idioma del audio (ISO 639-1). Vacío = autodetectar.    | `es`                                           |

- En **macOS**, `WHISPER_MODEL` es el repo del modelo MLX local dentro de `~/AI`
  (por ejemplo `mlx-community/whisper-large-v3-turbo`).
- En **Windows/Linux**, es un tamaño de faster-whisper (`tiny`, `base`, `small`,
  `medium`, `large-v3`) que se descarga automáticamente.

## Estructura del proyecto

```
main.py             Script principal (grabación, transcripción, pegado).
requirements.txt    Dependencias (el motor de Whisper depende del sistema operativo).
.env.example        Plantilla de configuración.
```

## Notas

- La tecla de activación es F5. Se puede cambiar editando `HOTKEY` en `main.py`.
- Whisper trabaja a 16 kHz mono; el audio del micrófono se captura a esa frecuencia.
