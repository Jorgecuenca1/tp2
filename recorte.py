import librosa
import librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd
import ipywidgets as widgets
from IPython.display import display, clear_output
import numpy as np

def recortar(x, fs):
    duration = len(x)
    markers = []

    # --- Funciones internas ---
    def plot_signal(t_start, t_end):
        plt.figure(figsize=(12, 4))
        times = np.arange(len(x))
        plt.plot(times, x, alpha=0.7)
        plt.axvline(x=t_start, color='r', label='Inicio')
        plt.axvline(x=t_end, color='b', label='Fin')
        plt.xlim(0, duration)
        plt.xlabel('Muestra')
        plt.ylabel('Amplitud')
        plt.title('Forma de onda con líneas de segmento')
        plt.legend()
        plt.show()

    def update_plot(_=None):
        t_start = start_slider.value
        t_end = end_slider.value
        if t_end < t_start:
            t_end = t_start
            end_slider.value = t_end
        with output:
            clear_output(wait=True)
            plot_signal(t_start, t_end)

    def play_segment(b):
        t_start = start_slider.value
        t_end = end_slider.value
        if t_end < t_start:
            t_end = t_start
        start_sample = int(t_start)
        end_sample = int(t_end)
        with output:
            clear_output(wait=True)
            plot_signal(t_start, t_end)
            display(ipd.Audio(x[start_sample:end_sample], rate=fs))

    def add_marker(b):
        t_start = start_slider.value
        t_end = end_slider.value
        if t_end < t_start:
            t_end = t_start
        markers.append((t_start, t_end))
        with output:
            clear_output(wait=True)
            plot_signal(t_start, t_end)
            print("Marcadores guardados (muestras):")
            for i, (s, e) in enumerate(markers):
                print(f"{i+1}: {s:.0f} - {e:.0f}")

    # --- Widgets ---
    start_slider = widgets.FloatSlider(
        value=5000, min=5000, max=duration - 5000, step=100,
        description='Inicio:', continuous_update=True
    )
    end_slider = widgets.FloatSlider(
        value=6000, min=5000, max=duration - 5000, step=100,
        description='Fin:', continuous_update=True
    )
    play_button = widgets.Button(description="Reproducir segmento")
    add_marker_button = widgets.Button(description="Agregar marcador")

    output = widgets.Output()

    # --- Eventos ---
    start_slider.observe(update_plot, names='value')
    end_slider.observe(update_plot, names='value')
    play_button.on_click(play_segment)
    add_marker_button.on_click(add_marker)

    # --- Mostrar interfaz ---
    display(start_slider, end_slider, play_button, add_marker_button, output)
    with output:
        plot_signal(start_slider.value, end_slider.value)

    return markers
