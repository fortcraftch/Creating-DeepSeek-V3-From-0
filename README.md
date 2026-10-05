# Creating-DeepSeek-V3-From-0
Recreating the deepseek v3 arquitecture for studie and comparison with other LLM's


## Entrenamiento con recuperación de errores CUDA

Desde la raíz de este proyecto, para una ejecución nueva:

```bash
python -u train_deepseek_v3.py --auto-restart --max-restarts 5 --restart-delay 15 --save-interval 25 --keep-checkpoints 3 --log-dir log_restart
```

Para continuar un checkpoint existente, añadir `--resume RUTA/model_XXXXX.pt`.
Ya no se intenta cargar automáticamente `log/model_00250.pt`. Mantener el mismo
batch, longitud, presupuesto de tokens y calendario al reanudar. `--max-steps`
es el paso final de la ejecución, no una cantidad adicional de pasos.
`--data-root` permite seleccionar los shards; por defecto `edu_fineweb10B`.
`--skip-hellaswag --skip-sampling` omite esas comprobaciones auxiliares, pero
mantiene la validación de pérdida. El modelo/configuración de pesos no se modifica.

El supervisor inicia un proceso nuevo tras un error CUDA reconocido y carga el
último checkpoint publicado por esa ejecución (pesos, optimizador, posición de
datos y RNG). Hace como máximo cinco reintentos; Ctrl+C y errores no CUDA no se
reinician. Si todavía no había checkpoint, vuelve al estado inicial. Esto no
repara el driver ni garantiza recuperación si el dispositivo sigue bloqueado.

Se guarda cada 25 actualizaciones, y al finalizar, mediante temporal, fsync y
reemplazo atómico. Después se eliminan checkpoints `model_NUMERO.pt` de la carpeta
de salida, conservando los tres más recientes y protegiendo el recién guardado.
`--keep-checkpoints 0` conserva todos. Usar una carpeta de salida específica del
experimento. No se reinicia desde archivos temporales ni se captura un checkpoint
nuevo dentro de un contexto CUDA que ya ha fallado.

Los checkpoints nuevos se guardan DESPUÉS de optimizer.step. Los históricos de
este script se guardaban ANTES del paso numerado: al reanudarlos se ejecuta ese
paso, evitando saltarse una actualización. Los antiguos no tienen RNG completo.
La pérdida guardada identifica el paso de validación y corresponde a antes de
su actualización; puede preceder al checkpoint si se guarda con más frecuencia.

`--config archivo.json` define una arquitectura para una ejecución nueva.
`--help` muestra batch, secuencia, validación y calendario configurables.

Validación local: seis pruebas del supervisor (CUDA simulado, errores no CUDA,
interrupción, retención y escritura atómica) y un modelo diminuto CPU con pesos,
cursor de datos y RNG idénticos al reanudar. No se ha provocado un timeout real
en homelab ni se ha lanzado allí un entrenamiento largo.

```bash
python -m unittest discover -s tests -p test_train_restart.py
```
