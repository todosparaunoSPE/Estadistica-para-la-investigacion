# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 17:07:50 2024

@author: jperezr
"""


import streamlit as st
from scipy.stats import multinomial
from scipy.special import comb
from scipy.stats import binom
from scipy.stats import poisson

import math
import scipy.stats as stats

import numpy as np

# Configurar el sidebar
st.sidebar.title("Selecciona un ejercicio")
exercise = st.sidebar.selectbox("Elige el ejercicio que quieres visualizar:", 
                                ("Ejercicio 1", 
                                 "Ejercicio 2",
                                 "Ejercicio 3",
                                 "Ejercicio 4",
                                 "Ejercicio 5",
                                 "Ejercicio 6",
                                 "Ejercicio 7",
                                 "Ejercicio 8",
                                 "Ejercicio 9",
                                 "Ejercicio 10",
                                 "Ejercicio 11",
                                 "Ejercicio 12"))

if exercise == "Ejercicio 1":
    # Título del ejercicio
    st.title("Esperanza Matemática en el Uso de Teléfonos en Clase")

    # Descripción del ejercicio
    st.write("""
    En un salón de clases hay 48 alumnos. En un determinado momento, se observa el grupo y algunos alumnos están usando su teléfono en clase. Sea X el número de alumnos usando su teléfono, determina E(X).
    """)

    # Interacción con el usuario para ingresar la probabilidad
    prob = st.slider("Selecciona la probabilidad de que un alumno esté usando su teléfono:", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    # Número de alumnos en el salón
    n_alumnos = 48

    # Cálculo de la esperanza matemática
    E_X = n_alumnos * prob

    # Mostrar el cálculo paso a paso
    st.write("### Cálculo del valor esperado:")
    st.latex(r'''
    E(X) = n \cdot p
    ''')
    st.write(f"Donde:")
    st.write(f"- \( n = {n_alumnos} \): Número de alumnos")
    st.write(f"- \( p = {prob:.2f} \): Probabilidad de que un alumno esté usando su teléfono")

    # Mostrar el resultado
    st.latex(r'''
    E(X) = 48 \cdot {:.2f} = {:.2f}
    '''.format(prob, E_X))

    st.write(f"Por lo tanto, la esperanza matemática E(X) es {E_X:.2f} alumnos usando su teléfono en clase.")

elif exercise == "Ejercicio 2":
    # Título del ejercicio
    st.title("Probabilidad de Acuerdos Alcanzados en una Junta")

    # Descripción del ejercicio
    st.write("""
    El gerente de cierta área considera que los colaboradores A, B y C tienen similares aptitudes para ser líderes de proyecto. Así, determina que lideren las juntas el mismo número de minutos cada reunión. 
    Se sabe que el 40% de los acuerdos alcanzados son de C, mientras que A y B consiguen un 30% de acuerdos, respectivamente. 
    Calcular la probabilidad de que en una junta con 9 acuerdos alcanzados, A consiguiera dos, B tres y C los 4 restantes.
    """)

    # Interacción con el usuario para ingresar los acuerdos alcanzados
    acuerdos_A = st.number_input("Ingrese el número de acuerdos alcanzados por A:", min_value=0, max_value=9, value=2, step=1)
    acuerdos_B = st.number_input("Ingrese el número de acuerdos alcanzados por B:", min_value=0, max_value=9, value=3, step=1)
    acuerdos_C = st.number_input("Ingrese el número de acuerdos alcanzados por C:", min_value=0, max_value=9, value=4, step=1)

    # Interacción con el usuario para ingresar las probabilidades
    prob_A = st.slider("Seleccione la probabilidad de que A consiga un acuerdo:", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
    prob_B = st.slider("Seleccione la probabilidad de que B consiga un acuerdo:", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
    prob_C = st.slider("Seleccione la probabilidad de que C consiga un acuerdo:", min_value=0.0, max_value=1.0, value=0.4, step=0.01)

    # Validar que la suma de acuerdos y probabilidades sea correcta
    total_acuerdos = acuerdos_A + acuerdos_B + acuerdos_C
    total_probabilidades = prob_A + prob_B + prob_C

    if total_acuerdos == 9 and total_probabilidades == 1.0:
        # Parámetros del problema
        acuerdos = [acuerdos_A, acuerdos_B, acuerdos_C]
        probabilidades = [prob_A, prob_B, prob_C]

        # Cálculo de la probabilidad usando distribución multinomial
        probabilidad = multinomial.pmf(acuerdos, total_acuerdos, probabilidades)

        # Mostrar el cálculo paso a paso
        st.write("### Cálculo de la probabilidad:")
        st.latex(r'''
        P(X_A = {}, X_B = {}, X_C = {}) = \frac{{9!}}{{{}!{}!{}!}} \cdot ({:.2f})^{{}} \cdot ({:.2f})^{{}} \cdot ({:.2f})^{{}}
        '''.format(acuerdos_A, acuerdos_B, acuerdos_C, acuerdos_A, acuerdos_B, acuerdos_C, prob_A, prob_B, prob_C))

        # Mostrar el resultado
        st.write(f"Por lo tanto, la probabilidad de que A consiga {acuerdos_A} acuerdos, B consiga {acuerdos_B} acuerdos y C consiga {acuerdos_C} acuerdos es {probabilidad:.5f}.")
    else:
        st.write("La suma de los acuerdos debe ser 9 y la suma de las probabilidades debe ser 1.0. Ajuste los valores y vuelva a intentarlo.")

elif exercise == "Ejercicio 3":
    # Título del ejercicio
    st.title("Probabilidad de Representantes en una Comisión")

    # Descripción del ejercicio
    st.write("""
    En un equipo de trabajo con 12 integrantes, han hecho una comisión de 4 representantes. En la plantilla hay 3 asesores, 3 programadores y 6 técnicos. ¿Cuál es la probabilidad de que haya 2 asesores y 2 programadores?
    """)

    # Interacción con el usuario para ingresar los representantes seleccionados
    asesores = st.number_input("Ingrese el número de asesores seleccionados:", min_value=0, max_value=4, value=2, step=1)
    programadores = st.number_input("Ingrese el número de programadores seleccionados:", min_value=0, max_value=4, value=2, step=1)
    tecnicos = st.number_input("Ingrese el número de técnicos seleccionados:", min_value=0, max_value=4, value=0, step=1)

    # Validar que la suma de representantes seleccionados sea correcta
    total_seleccionados = asesores + programadores + tecnicos

    if total_seleccionados == 4:
        # Parámetros del problema
        n_asesores = 3
        n_programadores = 3
        n_tecnicos = 6
        n_total = n_asesores + n_programadores + n_tecnicos

        # Cálculo de la probabilidad usando combinaciones
        prob_asesores = comb(n_asesores, asesores)
        prob_programadores = comb(n_programadores, programadores)
        prob_tecnicos = comb(n_tecnicos, tecnicos)
        total_combinaciones = comb(n_total, total_seleccionados)

        probabilidad = (prob_asesores * prob_programadores * prob_tecnicos) / total_combinaciones

        # Mostrar el cálculo paso a paso
        st.write("### Cálculo de la probabilidad:")
        st.latex(r'''
        P(\text{{2 asesores, 2 programadores}}) = \frac{{C(3, 2) \cdot C(3, 2) \cdot C(6, 0)}}{{C(12, 4)}}
        ''')

        # Mostrar el resultado
        st.write(f"Por lo tanto, la probabilidad de que haya 2 asesores y 2 programadores en la comisión es {probabilidad:.5f}.")
    else:
        st.write("La suma de los representantes seleccionados debe ser 4. Ajuste los valores y vuelva a intentarlo.")
        
elif exercise == "Ejercicio 4":
    # Título del ejercicio
    st.title("Probabilidad de Jugar al Menos 4 Partidos en un Torneo de Básquetbol")

    # Descripción del ejercicio
    st.write("""
    En un torneo escolar de básquetbol, cierto equipo tiene una probabilidad de 60% de ganar un partido. Si el equipo juega hasta perder un partido, encuentra la probabilidad de que juegue al menos 4 partidos.
    """)

    # Interacción con el usuario para ingresar la probabilidad de ganar
    prob_ganar = st.slider("Selecciona la probabilidad de ganar un partido:", min_value=0.0, max_value=1.0, value=0.6, step=0.01)

    # Cálculo de la probabilidad de jugar al menos 4 partidos
    prob_perder = 1 - prob_ganar
    probabilidad_4_partidos = prob_ganar**3 * prob_perder

    # Mostrar el cálculo paso a paso
    st.write("### Cálculo de la probabilidad:")
    st.latex(r'''
    P(\text{{jugar al menos 4 partidos}}) = p^3 \cdot (1 - p)
    ''')
    st.write(f"Donde:")
    st.write(f"- \( p = {prob_ganar:.2f} \): Probabilidad de ganar un partido")

    # Mostrar el resultado
    st.latex(r'''
    P(\text{{jugar al menos 4 partidos}}) = {:.2f}^3 \cdot (1 - {:.2f}) = {:.5f}
    '''.format(prob_ganar, prob_ganar, probabilidad_4_partidos))

    st.write(f"Por lo tanto, la probabilidad de que el equipo juegue al menos 4 partidos es {probabilidad_4_partidos:.5f}.")
    
    
    
    
if exercise == "Ejercicio 5":
    # Título del ejercicio
    st.title("Probabilidad de Usar un Cajón de Estacionamiento")

    # Descripción del ejercicio
    st.write("""
    El cajón de estacionamiento más cercano a tu área de trabajo está desocupado el 15% del tiempo. 
    Si debes hacer uso del cajón en 5 ocasiones distintas durante el mes y las ocasiones son independientes entre sí, 
    determina la probabilidad de que:
    a. El cajón esté desocupado todas las veces que acudes a él.
    b. El cajón esté desocupado 4 de las 5 veces en que acudes a él.
    c. El cajón esté desocupado al menos 3 veces de las 5 que vayas a usarlo.
    """)

    # Parámetros del problema
    prob_desocupado = 0.15  # Probabilidad de que el cajón esté desocupado
    n_occasions = 5  # Número de ocasiones en las que se usará el cajón

    # Interacción con el usuario para seleccionar el caso
    case = st.selectbox("Selecciona el caso que deseas calcular:", ["a", "b", "c"])

    if case == "a":
        # Cálculo de la probabilidad de que el cajón esté desocupado todas las veces
        prob_all_desocupado = binom.pmf(5, n_occasions, prob_desocupado)
        st.write("### Cálculo de la probabilidad de que el cajón esté desocupado todas las veces:")
        st.latex(r'''
        P(X = 5) = \binom{5}{5} \cdot (0.15)^5 \cdot (0.85)^0
        ''')
        st.write(f"Por lo tanto, la probabilidad de que el cajón esté desocupado todas las veces es {prob_all_desocupado:.5f}.")

    elif case == "b":
        # Cálculo de la probabilidad de que el cajón esté desocupado 4 de las 5 veces
        prob_4_desocupado = binom.pmf(4, n_occasions, prob_desocupado)
        st.write("### Cálculo de la probabilidad de que el cajón esté desocupado 4 de las 5 veces:")
        st.latex(r'''
        P(X = 4) = \binom{5}{4} \cdot (0.15)^4 \cdot (0.85)^1
        ''')
        st.write(f"Por lo tanto, la probabilidad de que el cajón esté desocupado 4 de las 5 veces es {prob_4_desocupado:.5f}.")

    elif case == "c":
        # Cálculo de la probabilidad de que el cajón esté desocupado al menos 3 de las 5 veces
        prob_at_least_3_desocupado = binom.sf(2, n_occasions, prob_desocupado)  # 1 - P(X <= 2)
        st.write("### Cálculo de la probabilidad de que el cajón esté desocupado al menos 3 de las 5 veces:")
        st.latex(r'''
        P(X \geq 3) = 1 - \sum_{k=0}^{2} \binom{5}{k} \cdot (0.15)^k \cdot (0.85)^{5-k}
        ''')
        st.write(f"Por lo tanto, la probabilidad de que el cajón esté desocupado al menos 3 de las 5 veces es {prob_at_least_3_desocupado:.5f}.")
        
        
        
elif exercise == "Ejercicio 6":
    # Título del ejercicio
    st.title("Probabilidad de Conductores en un Viaje")

    # Descripción del ejercicio
    st.write("""
    Supón que el número de conductores que viajan entre CDMX y Acapulco durante un
    periodo designado tiene una distribución de Poisson con parámetro λ. 
    Determina la probabilidad de que el número de conductores sea más de 20.
    """)

    # Interacción con el usuario para ingresar el parámetro λ
    lambda_param = st.slider("Selecciona el valor del parámetro λ:", min_value=1, max_value=50, value=20, step=1)

    # Cálculo de la probabilidad de que el número de conductores sea mayor de 20
    prob_mayor_20 = 1 - poisson.cdf(20, lambda_param)  # 1 - P(X <= 20)

    # Mostrar el cálculo paso a paso
    st.write("### Cálculo de la probabilidad:")
    st.latex(r'''
    P(X > 20) = 1 - P(X \leq 20) = 1 - \sum_{k=0}^{20} \frac{\lambda^k e^{-\lambda}}{k!}
    ''')
    st.write(f"Donde:")
    st.write(f"- \( \lambda = {lambda_param} \): Parámetro de la distribución de Poisson")

    # Mostrar el resultado
    st.write(f"### Resultado:")
    st.latex(r'''
    P(X > 20) = 1 - \sum_{k=0}^{20} \frac{{{lambda_param}^k \cdot e^{-{lambda_param}}}}{k!}
    ''')
    st.write(f"Por lo tanto, la probabilidad de que el número de conductores sea más de 20 es {prob_mayor_20:.5f}.")
    
    
    
elif exercise == "Ejercicio 7":
    # Título del ejercicio
    st.title("Probabilidad de que la Primera Llamada Exitosa sea la Décima")

    # Descripción del ejercicio
    st.write("""
    Supón que cada una de las llamadas que hace una persona a una línea de quejas y sugerencias de tarjetas de crédito tiene una probabilidad de 0.02 de que la línea no esté ocupada. 
    Supón que las llamadas son independientes. ¿Cuál es la probabilidad de que la primera llamada exitosa sea la décima que realiza la persona?
    """)

    # Interacción con el usuario para ingresar la probabilidad de que la línea no esté ocupada
    prob_no_ocupada = st.slider("Selecciona la probabilidad de que la línea no esté ocupada:", min_value=0.0, max_value=1.0, value=0.02, step=0.01)

    # Número de llamadas realizadas antes de la primera llamada exitosa
    n_llamadas = st.number_input("Ingrese el número de llamadas realizadas antes de la primera llamada exitosa:", min_value=10, max_value=50, value=10, step=1)

    # Validar que el número de llamadas sea mayor o igual a 10
    if n_llamadas >= 10:
        # Cálculo de la probabilidad usando la distribución geométrica
        probabilidad = (1 - prob_no_ocupada)**(n_llamadas - 1) * prob_no_ocupada

        # Mostrar el cálculo paso a paso
        st.write("### Cálculo de la probabilidad:")
        st.latex(r'''
        P(\text{{primera llamada exitosa sea la décima}}) = (1 - p)^{n-1} \cdot p
        ''')
        st.write(f"Donde:")
        st.write(f"- \( p = {prob_no_ocupada:.2f} \): Probabilidad de que la línea no esté ocupada")
        st.write(f"- \( n = {n_llamadas} \): Número de llamadas realizadas")

        # Mostrar el resultado
        st.latex(r'''
        P(\text{{primera llamada exitosa sea la décima}}) = (1 - {:.2f})^{{{}}} \cdot {:.2f} = {:.5f}
        '''.format(prob_no_ocupada, n_llamadas - 1, prob_no_ocupada, probabilidad))

        st.write(f"Por lo tanto, la probabilidad de que la primera llamada exitosa sea la {n_llamadas}ª llamada es {probabilidad:.5f}.")
    else:
        st.write("El número de llamadas realizadas antes de la primera llamada exitosa debe ser al menos 10. Ajuste el valor y vuelva a intentarlo.")    
        
        
        

elif exercise == "Ejercicio 8":
    
    # Título y descripción del ejercicio
    st.title("Determinación del Intervalo de una Variable Aleatoria Uniforme con Varianza Conocida")
    # Mostrar el enunciado del ejercicio
    st.write("**Ejercicio 8:**")
    st.write("Supón que X es una variable aleatoria que se distribuye uniformemente de manera"
         " simétrica respecto al cero y con varianza 1. Obtén los valores apropiados para el"
         " intervalo en que existe X.")

    # Paso 1: Ingreso de la varianza por el usuario
    st.write("### Paso 1: Ingresar la varianza")
    varianza_input = st.number_input("Ingresa la varianza (σ²):", value=1.0, step=0.01)

    # Explicación sobre el cálculo de b
    st.write("### Explicación del cálculo de 'b'")
    st.write("""
    Para una variable aleatoria \(X\) uniformemente distribuida de manera simétrica respecto al cero, 
    el intervalo es \([-b, b]\). La fórmula para calcular la varianza de \(X\) es:

    \[ \text{Var}(X) = \frac{(b - (-b))^2}{12} \]

    Simplificando la expresión:

    \[ \text{Var}(X) = \frac{(2b)^2}{12} = \frac{4b^2}{12} = \frac{b^2}{3} \]

    Despejamos \(b\) de la ecuación \(\text{Var}(X) = \frac{b^2}{3}\):

    \[ \text{Var}(X) \times 3 = b^2 \]

    \[ b = \sqrt{3 \cdot \text{Var}(X)} \]

    En este ejercicio, usamos una varianza conocida de 1, pero permitimos al usuario ingresar diferentes valores de varianza.
    """)

    # Paso 2: Cálculo del valor de 'b' a partir de la varianza
    st.write("### Paso 2: Cálculo del valor de 'b'")

    if 'b' not in st.session_state:
        st.session_state.b = None

    if st.button("Calcular 'b'"):
       st.session_state.b = math.sqrt(varianza_input * 3)
       st.write(f"El valor calculado de 'b' es: {st.session_state.b:.2f}")

    # Paso 3: Determinación del intervalo
    st.write("### Paso 3: Determinación del intervalo")

    if st.session_state.b is not None:
        if st.button("Determinar intervalo"):
            intervalo = f"[{-st.session_state.b:.2f}, {st.session_state.b:.2f}]"
            st.write(f"El intervalo en el que X existe es: {intervalo}")
        
            # Mostrar todos los pasos y resultados juntos
            st.write("### Resumen:")
            st.write(f"Varianza ingresada: {varianza_input}")
            st.write(f"Valor de 'b': {st.session_state.b:.2f}")
            st.write(f"Intervalo: {intervalo}")
        else:
            st.write("Por favor, calcula 'b' primero.")





    
    
if exercise == "Ejercicio 9":
    # Título del ejercicio
    st.title("Porcentaje de Garantías que Tendrán que Hacerse Efectivas")

    # Descripción del ejercicio
    st.write("""
    Se estima que el tiempo transcurrido hasta la falla de una pieza mecánica se distribuye exponencialmente con una media de tres años. 
    Una compañía ofrece garantía por el primer año de uso. ¿Qué porcentaje de garantías tendrá que hacer efectivas por este tipo de fallas?
    """)

    # Parámetros del problema
    tiempo_medio = 3  # Tiempo medio hasta la falla en años
    duracion_garantia = 1  # Duración de la garantía en años

    # Interacción con el usuario para ingresar los parámetros
    tiempo_medio = st.number_input("Ingrese el tiempo medio hasta la falla (en años):", value=3.0)
    duracion_garantia = st.number_input("Ingrese la duración de la garantía (en años):", value=1.0)

    # Calcular el porcentaje de garantías
    tasa_fallas = 1 / tiempo_medio
    probabilidad_falla = stats.expon.cdf(duracion_garantia, scale=1/tasa_fallas)
    porcentaje_garantias = probabilidad_falla * 100

    # Mostrar resultados
    st.write("### Cálculo del porcentaje de garantías que tendrán que hacerse efectivas:")
    st.latex(r'''
    P(\text{falla en el primer año}) = 1 - e^{-\lambda t}
    ''')
    st.write(f"Donde la tasa de fallas es λ = {tasa_fallas:.2f} y el porcentaje de garantías que tendrán que hacerse efectivas es {porcentaje_garantias:.2f}%.")
    
    
    
    
    
if exercise == "Ejercicio 10":
    # Título del ejercicio
    st.title("Probabilidad de Envío Dentro de un Rango de Días")

    # Descripción del ejercicio
    st.write("""
    El tiempo de reabastecimiento de cierto producto cumple con la distribución gamma. 
    Puedes ajustar los parámetros de la distribución gamma (media y varianza) y los días de interés para determinar la probabilidad de que un pedido se envíe dentro de un rango de días específico.
    """)

    # Entrada del usuario para la media y la varianza
    media = st.number_input("Ingrese la media de la distribución (en días):", value=40)
    varianza = st.number_input("Ingrese la varianza de la distribución (en días^2):", value=400)

    # Entrada del usuario para los días de interés
    t1 = st.number_input("Ingrese el número de días mínimo para el envío:", value=20)
    t2 = st.number_input("Ingrese el número de días máximo para el envío:", value=60)

    # Validar que t2 sea mayor que t1
    if t2 <= t1:
        st.error("El número de días máximo debe ser mayor que el número de días mínimo.")
    else:
        # Calcular los parámetros de la distribución gamma
        theta = varianza / media
        k = (media ** 2) / varianza

        # Calcular las probabilidades
        prob_antes_de_t2 = stats.gamma.cdf(t2, a=k, scale=theta)
        prob_antes_de_t1 = stats.gamma.cdf(t1, a=k, scale=theta)
        prob_entre_t1_y_t2 = prob_antes_de_t2 - prob_antes_de_t1

        # Mostrar resultados
        st.write("### Cálculo de la probabilidad de que el pedido se envíe dentro del rango de días:")
        st.latex(r'''
        \text{Para la distribución gamma con } \mu = \text{media} \text{ y } \sigma^2 = \text{varianza}:
        ''')
        st.latex(r'''
        \text{Parámetros: } k = \frac{\mu^2}{\sigma^2} \text{ y } \theta = \frac{\sigma^2}{\mu}
        ''')
        st.write(f"Parámetro de forma (k): {k:.2f}")
        st.write(f"Parámetro de escala (θ): {theta:.2f}")
        st.latex(r'''
        \text{Probabilidad de envío dentro de } t_2 \text{ días: } P(X \leq t_2) = F(t_2)
        ''')
        st.latex(r'''
        \text{Probabilidad de envío dentro de } t_1 \text{ días: } P(X \leq t_1) = F(t_1)
        ''')
        st.write(f"Probabilidad de que el pedido se envíe dentro de {t2} días: {prob_antes_de_t2:.4f}")
        st.write(f"Probabilidad de que el pedido se envíe dentro de {t1} días: {prob_antes_de_t1:.4f}")
        st.write(f"Probabilidad de que el pedido se envíe entre {t1} y {t2} días: {prob_entre_t1_y_t2:.4f}")
        
        
        
if exercise == "Ejercicio 11":
    # Título del ejercicio
    st.title("Probabilidad de Diámetro en la Distribución de Weibull")

    # Descripción del ejercicio
    st.write("""
    El diámetro de unos ejes de acero sigue una distribución de Weibull con los siguientes parámetros:
    - \(\gamma\) = 1.0 pulgadas (escala)
    - \(\beta\) = 2 (forma)
    - \(\delta\) = 0.5 pulgadas (ubicación)
    
    Calcula la probabilidad de que un eje seleccionado al azar no exceda el diámetro especificado.
    """)

    # Parámetros fijos para la distribución Weibull
    beta = 2.0  # Parámetro de forma
    gamma = 1.0  # Parámetro de escala (en pulgadas)
    delta = 0.5  # Parámetro de ubicación (en pulgadas)

    # Entrada del usuario para el diámetro máximo
    max_diameter = st.number_input("Ingrese el diámetro máximo (en pulgadas):", value=1.5)

    # Validar que el diámetro máximo sea mayor que el parámetro de ubicación
    if max_diameter <= delta:
        st.error("El diámetro máximo debe ser mayor que el parámetro de ubicación (δ).")
    else:
        # Cálculo de la probabilidad usando la fórmula Weibull correcta
        term = (max_diameter - delta) / gamma
        prob_no_excede = 1 - np.exp(-term ** beta)

        # Mostrar resultados
        st.write("### Cálculo de la probabilidad de que el diámetro no exceda el valor especificado:")
        st.latex(r'''
        \text{Para la distribución Weibull con parámetros } \beta, \gamma, \text{ y } \delta:
        ''')
        st.latex(r'''
        \text{Probabilidad de que el diámetro no exceda } D \text{ pulgadas es } P(X \leq D) = 1 - \exp\left(-\left(\frac{D - \delta}{\gamma}\right)^\beta\right)
        ''')
        st.write(f"Parámetro de forma (β): {beta:.2f}")
        st.write(f"Parámetro de escala (γ): {gamma:.2f}")
        st.write(f"Parámetro de ubicación (δ): {delta:.2f}")
        st.write(f"Diámetro máximo considerado: {max_diameter:.2f} pulgadas")
        st.write(f"Probabilidad de que el diámetro no exceda {max_diameter} pulgadas: {prob_no_excede:.8f}")
        
        
        
if exercise == "Ejercicio 12":
    # Título del ejercicio
    st.title("Vida Útil de una Batería")

    # Descripción del ejercicio
    st.write("""
    La vida útil de una batería de celda seca se distribuye normalmente con una media de 600 días y una desviación estándar de 60 días.
    
    a. ¿Qué fracción de estas baterías se esperaría que durara más de 680 días?
    
    b. ¿Qué fracción de estas baterías se esperaría que fallara antes de 560 días?
    """)

    # Parámetros de la distribución normal
    mean = 600  # Media de la vida útil (en días)
    std_dev = 60  # Desviación estándar de la vida útil (en días)

    # Interacción con el usuario para seleccionar el caso
    case = st.selectbox("Selecciona el cálculo que deseas realizar:", ["a", "b"])

    if case == "a":
        # Entrada del usuario para el número de días
        threshold = st.number_input("Ingrese el umbral de días (para duración mayor a este valor):", value=680)

        # Cálculo de la probabilidad de que la batería dure más de `threshold` días
        prob_more_than_threshold = 1 - stats.norm.cdf(threshold, loc=mean, scale=std_dev)

        # Mostrar resultados
        st.write("### Cálculo de la fracción de baterías que durarán más de los días especificados:")
        st.latex(r'''
        P(X > \text{umbral}) = 1 - P(X \leq \text{umbral})
        ''')
        st.write(f"Media de la vida útil: {mean:.2f} días")
        st.write(f"Desviación estándar: {std_dev:.2f} días")
        st.write(f"Umbral de días: {threshold:.2f} días")
        st.write(f"Fracción de baterías que durarán más de {threshold} días: {prob_more_than_threshold:.8f}")

    elif case == "b":
        # Entrada del usuario para el número de días
        threshold = st.number_input("Ingrese el umbral de días (para duración menor a este valor):", value=560)

        # Cálculo de la probabilidad de que la batería falle antes de `threshold` días
        prob_less_than_threshold = stats.norm.cdf(threshold, loc=mean, scale=std_dev)

        # Mostrar resultados
        st.write("### Cálculo de la fracción de baterías que fallarán antes de los días especificados:")
        st.latex(r'''
        P(X < \text{umbral}) = P(X \leq \text{umbral})
        ''')
        st.write(f"Media de la vida útil: {mean:.2f} días")
        st.write(f"Desviación estándar: {std_dev:.2f} días")
        st.write(f"Umbral de días: {threshold:.2f} días")
        st.write(f"Fracción de baterías que fallarán antes de {threshold} días: {prob_less_than_threshold:.8f}")        
        
        
st.sidebar.title("Estadística para la Investigación")
st.sidebar.write("Primera sesión de trabajo personal")        
st.sidebar.write("Javier Horacio Pérez Ricárdez")        
st.sidebar.write("© 2024 Todos los derechos reservados")
