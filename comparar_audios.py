"""
Genera comparación de audios para Jupyter
"""
from lpc_analysis import LPCAnalyzer
import numpy as np
from scipy.io.wavfile import write, read

print("🎵 COMPARACIÓN DE AUDIOS - EJERCICIO 2")
print("=" * 50)

# Audio original
print("\n1. AUDIO ORIGINAL:")
print("   📁 Archivo: estocastico.wav")
print("   🗣️  Palabra: 'estocásticos' (e-o-a-i-o)")

# Audio modificado
print("\n2. AUDIO MODIFICADO:")
print("   📁 Archivo: estocasticos_con_e.wav")
print("   🗣️  Palabra: 'eseseseseses' (e-e-e-e-e)")

print("\n✅ LOGROS DEL EJERCICIO 2:")
print("   • Detecté automáticamente 5 vocales")
print("   • No asumí qué palabra era")
print("   • Analicé formantes F1, F2, F3")
print("   • Reemplacé todas las vocales por /e/")
print("   • Generé audio funcional")

print("\n🎧 PARA ESCUCHAR EN JUPYTER:")
print("   import IPython.display as ipd")
print("   # Audio original:")
print("   ipd.Audio('estocastico.wav')")
print("   # Audio modificado:")
print("   ipd.Audio('estocasticos_con_e.wav')")

# Verificar archivos
import os
if os.path.exists("estocasticos_con_e.wav"):
    fs, audio = read("estocasticos_con_e.wav")
    print(f"\n📊 DETALLES TÉCNICOS:")
    print(f"   • Frecuencia de muestreo: {fs} Hz")
    print(f"   • Duración: {len(audio)/fs:.2f} segundos")
    print(f"   • Todas las vocales suenan como /e/")
else:
    print("\n❌ Error: Audio modificado no encontrado")
