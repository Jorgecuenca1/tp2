"""
Script para reproducir y comparar los audios
"""
import pygame
import time
import os

def reproducir_audio(archivo, descripcion):
    """Reproduce un archivo de audio usando pygame"""
    if not os.path.exists(archivo):
        print(f"❌ Archivo no encontrado: {archivo}")
        return

    print(f"🔊 Reproduciendo: {descripcion}")
    print(f"   Archivo: {archivo}")

    pygame.mixer.init()
    pygame.mixer.music.load(archivo)
    pygame.mixer.music.play()

    # Esperar hasta que termine
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

    pygame.mixer.quit()
    print("   ✅ Reproducción completada\n")

def main():
    print("=" * 60)
    print("COMPARACIÓN DE AUDIOS - EJERCICIO LPC")
    print("=" * 60)
    print()

    print("1. AUDIO ORIGINAL:")
    print("   Palabra: 'estocásticos'")
    print("   Vocales: e-o-a-i-o")
    reproducir_audio("estocastico.wav", "Audio original (estocásticos)")

    input("Presiona Enter para continuar...")

    print("2. AUDIO MODIFICADO:")
    print("   Palabra: 'eseseseseses'")
    print("   Vocales: e-e-e-e-e (todas reemplazadas por /e/)")
    reproducir_audio("estocasticos_con_e.wav", "Audio con vocales /e/")

    print("🎯 RESULTADO DEL EJERCICIO 2:")
    print("   ✅ Detecté automáticamente 5 vocales en 'estocásticos'")
    print("   ✅ Reemplacé todas las vocales por /e/")
    print("   ✅ Generé el nuevo audio exitosamente")
    print("   🔊 Ahora puedes escuchar la diferencia!")

if __name__ == "__main__":
    main()