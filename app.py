import streamlit as st
import numpy as np
import scipy.stats as sp

st.title('Ejercicios de Estadística para la investigación')

exercise = st.sidebar.selectbox(
    'Selecciona un ejercicio:',
    [f'Ejercicio {i}' for i in range(1, 13)]
)

if exercise == 'Ejercicio 1':
    st.header('Ejercicio 1')
    st.write('**Enunciado:** En un salón de clases hay 48 alumnos. En un determinado momento, se observa el grupo y algunos alumnos están usando su teléfono en clase. Sea X=número de alumnos usando su teléfono, determina E(X).')
    st.write('**Distribución de probabilidad utilizada:** Distribución binomial')
    st.write('**Datos:**')
    st.write('- Número de ensayos (n): 48')
    st.write('- Probabilidad de éxito (p): p')
    
    n = 48
    p = 0.5
    expected_value = n * p
    
    st.write('**Fórmula:**')
    st.write('E(X) = n * p')
    st.write('**Sustitución de los datos en la fórmula:**')
    st.write(f'E(X) = {n} * p')
    st.write(f'**Resultado:** El valor esperado es: E(X) = 48 * p')

elif exercise == 'Ejercicio 2':
    st.header('Ejercicio 2')
    st.write('**Enunciado:** El gerente de cierta área considera que los colaboradores A, B y C tienen similares aptitudes para ser líderes de proyecto. Así, determina que lideren las juntas el mismo número de minutos cada reunión. Se sabe que el 40% de los acuerdos alcanzados son de C, mientras que A y B consiguen un 30% de acuerdos, respectivamente. Calcular la probabilidad de que en una junta con 9 acuerdos alcanzados, A consiguiera dos, B tres y C los 4 restantes.')
    st.write('**Distribución de probabilidad utilizada:** Distribución multinomial')
    st.write('**Datos:**')
    st.write('- Número total de acuerdos (n): 9')
    st.write('- Número de acuerdos para A (k_A): 2')
    st.write('- Número de acuerdos para B (k_B): 3')
    st.write('- Número de acuerdos para C (k_C): 4')
    st.write('- Probabilidad de acuerdo para A (p_A): 0.30')
    st.write('- Probabilidad de acuerdo para B (p_B): 0.30')
    st.write('- Probabilidad de acuerdo para C (p_C): 0.40')
    
    n = 9
    k_A = 2
    k_B = 3
    k_C = 4
    p_A = 0.30
    p_B = 0.30
    p_C = 0.40
    
    prob = sp.multinomial.pmf([k_A, k_B, k_C], n, [p_A, p_B, p_C])
    
    st.write('**Fórmula:**')
    st.write('P(X_A = k_A, X_B = k_B, X_C = k_C) = (n! / (k_A! * k_B! * k_C!)) * p_A^k_A * p_B^k_B * p_C^k_C')
    
    st.write('**Sustitución de los datos en la fórmula:**')
    st.write(f'P = (9! / (2! * 3! * 4!)) * 0.30^2 * 0.30^3 * 0.40^4')
    st.write(f'**Resultado:** La probabilidad es {prob:.4f}')

elif exercise == 'Ejercicio 3':
    st.header('Ejercicio 3')
    st.write('**Enunciado:** En un equipo de trabajo con 12 integrantes, han hecho una comisión de 4 representantes. En la plantilla hay 3 asesores, 3 programadores y 6 técnicos. ¿Cuál es la probabilidad de que haya 2 asesores y 2 programadores?')
    st.write('**Distribución de probabilidad utilizada:** Distribución hipergeométrica')
    st.write('**Datos:**')
    st.write('- Total de integrantes (N): 12')
    st.write('- Número de asesores (K_asesores): 3')
    st.write('- Número de programadores (K_programadores): 3')
    st.write('- Número de técnicos (K_tecnicos): 6')
    st.write('- Tamaño de la muestra (n): 4')
    st.write('- Número de asesores en la muestra (k_asesores): 2')
    st.write('- Número de programadores en la muestra (k_programadores): 2')
    
    N = 12
    K_asesores = 3
    K_programadores = 3
    K_tecnicos = 6
    n = 4
    k_asesores = 2
    k_programadores = 2
    
    prob = (sp.hypergeom.pmf(k_asesores, N, K_asesores, n) *
            sp.hypergeom.pmf(k_programadores, N - K_asesores, K_programadores, n - k_asesores))
    
    st.write('**Fórmula:**')
    st.write('P(X_asesores = k_asesores, X_programadores = k_programadores) = (C(K_asesores, k_asesores) * C(K_programadores, k_programadores)) / C(N, n)')
    
    st.write('**Sustitución de los datos en la fórmula:**')
    st.write(f'P = (C(3, 2) * C(3, 2)) / C(12, 4)')
    st.write(f'**Resultado:** La probabilidad es {prob:.4f}')

elif exercise == 'Ejercicio 4':
    st.header('Ejercicio 4')
    st.write('**Enunciado:** En un torneo escolar de básquetbol, cierto equipo tiene una probabilidad de 60% de ganar un partido. Si el equipo juega hasta perder un partido, encuentra la probabilidad de que juegue al menos 4 partidos.')
    st.write('**Distribución de probabilidad utilizada:** Distribución geométrica')
    st.write('**Datos:**')
    st.write('- Probabilidad de éxito (p): 0.60')
    st.write('- Número de partidos (k): 3 (para que juegue al menos 4 partidos, debe ganar los 3 primeros)')
    
    p = 0.60
    k = 3
    
    prob = (1 - p) ** k
    
    st.write('**Fórmula:**')
    st.write('P(X ≥ k + 1) = (1 - p)^k')
    
    st.write('**Sustitución de los datos en la fórmula:**')
    st.write(f'P(X ≥ 4) = (1 - {p})^{k}')
    st.write(f'**Resultado:** La probabilidad es {prob:.4f}')

elif exercise == 'Ejercicio 5':
    st.header('Ejercicio 5')
    st.write('**Enunciado:** El cajón de estacionamiento más cercano a tu área de trabajo está desocupado el 15% del tiempo. Si debes hacer uso del cajón en 5 ocasiones distintas durante el mes y las ocasiones son independientes entre sí, determina la probabilidad de que:')
    st.write('a. el cajón esté desocupado todas las veces que acudes a él.')
    st.write('b. el cajón esté desocupado 4 de las 5 veces en que acudes a él.')
    st.write('c. el cajón esté desocupado al menos 3 veces de las 5 que vayas a usarlo.')
    st.write('**Distribución de probabilidad utilizada:** Distribución binomial')
    st.write('**Datos:**')
    st.write('- Total de ocasiones (n): 5')
    st.write('- Probabilidad de éxito (p): 0.15')
    
    n = 5
    p = 0.15
    
    # a. Probabilidad de que esté desocupado todas las veces
    prob_all_empty = sp.binom.pmf(5, n, p)
    
    # b. Probabilidad de que esté desocupado 4 de las 5 veces
    prob_4_empty = sp.binom.pmf(4, n, p)
    
    # c. Probabilidad de que esté desocupado al menos 3 veces
    prob_at_least_3_empty = sum(sp.binom.pmf(k, n, p) for k in range(3, 6))
    
    st.write('**Fórmulas:**')
    st.write('a. P(X = 5) = C(n, 5) * p^5 * (1 - p)^0')
    st.write('b. P(X = 4) = C(n, 4) * p^4 * (1 - p)^1')
    st.write('c. P(X ≥ 3) = ∑ P(X = k) para k de 3 a 5')
    
    st.write('**Resultados:**')
    st.write(f'a. La probabilidad de que esté desocupado todas las veces es {prob_all_empty:.4f}')
    st.write(f'b. La probabilidad de que esté desocupado 4 de las 5 veces es {prob_4_empty:.4f}')
    st.write(f'c. La probabilidad de que esté desocupado al menos 3 veces es {prob_at_least_3_empty:.4f}')
    
   
if exercise == 'Ejercicio 6':
    st.header('Ejercicio 6')
    st.write('**Enunciado:** Supón que el número de conductores que viajan entre CDMX y Acapulco durante un periodo designado tiene una distribución de Poisson con parámetro λ=20. ¿Cuál es la probabilidad de que el número de conductores sea de más de 20?')
    st.write('**Distribución de probabilidad utilizada:** Distribución de Poisson')
    st.write('**Datos:**')
    st.write('- Parámetro de la distribución (λ): 20')
    
    lambda_ = 20
    k = 20
    prob = 1 - sp.poisson.cdf(k, lambda_)
    
    st.write('**Fórmula:**')
    st.write('P(X > k) = 1 - P(X ≤ k)')
    
    st.write('**Sustitución de los datos en la fórmula:**')
    st.write(f'P(X > {k}) = 1 - P(X ≤ {k})')
    st.write(f'**Resultado:** La probabilidad es {prob:.4f}')

elif exercise == 'Ejercicio 7':
    st.header('Ejercicio 7')
    st.write('**Enunciado:** Supón que cada una de las llamadas que hace una persona a una línea de quejas y sugerencias de tarjetas de crédito tiene una probabilidad de 0.02 de que la línea no esté ocupada. Supón que las llamadas son independientes. ¿Cuál es la probabilidad de que la primera llamada que entre sea la décima que realiza la persona?')
    st.write('**Distribución de probabilidad utilizada:** Distribución geométrica')
    st.write('**Datos:**')
    st.write('- Probabilidad de éxito (p): 0.02')
    st.write('- Número de llamadas (k): 10')
    
    p = 0.02
    k = 10
    prob = (1 - p) ** (k - 1) * p
    
    st.write('**Fórmula:**')
    st.write('P(X = k) = (1 - p)^(k - 1) * p')
    
    st.write('**Sustitución de los datos en la fórmula:**')
    st.write(f'P(X = {k}) = (1 - {p})^{k - 1} * {p}')
    st.write(f'**Resultado:** La probabilidad es {prob:.4f}')

elif exercise == 'Ejercicio 8':
    st.header('Ejercicio 8')
    st.write('**Enunciado:** Supón que X es una variable aleatoria que se distribuye uniformemente de manera simétrica respecto al cero y con varianza 1. Obtén los valores apropiados para el intervalo en que existe X.')
    st.write('**Distribución de probabilidad utilizada:** Distribución uniforme')
    st.write('**Datos:**')
    st.write('- Varianza (σ²): 1')
    
    variance = 1
    a = -np.sqrt(variance / 3)
    b = np.sqrt(variance / 3)
    
    st.write('**Fórmula:**')
    st.write('Varianza = (b - a)² / 12')
    
    st.write('**Sustitución de los datos en la fórmula:**')
    st.write(f'(b - a)² / 12 = {variance}')
    st.write(f'a = {-np.sqrt(variance / 3)}')
    st.write(f'b = {np.sqrt(variance / 3)}')
    st.write(f'**Intervalo:** X está en el intervalo [{a:.2f}, {b:.2f}]')

elif exercise == 'Ejercicio 9':
    st.header('Ejercicio 9')
    st.write('**Enunciado:** Se estima que el tiempo transcurrido hasta la falla de una pieza mecánica se distribuye exponencialmente con una media de tres años. Una compañía ofrece garantía por el primer año de uso. ¿Qué porcentaje de garantías tendrá que hacer efectivas por este tipo de fallas?')
    st.write('**Distribución de probabilidad utilizada:** Distribución exponencial')
    st.write('**Datos:**')
    st.write('- Media (μ): 3 años')
    st.write('- Tiempo de garantía: 1 año')
    
    mu = 3
    x = 1
    rate = 1 / mu
    prob = 1 - sp.expon.cdf(x, scale=mu)
    
    st.write('**Fórmula:**')
    st.write('P(X ≤ x) = 1 - e^(-rate * x)')
    
    st.write('**Sustitución de los datos en la fórmula:**')
    st.write(f'P(X ≤ {x}) = 1 - e^(-{rate} * {x})')
    st.write(f'**Resultado:** El porcentaje de garantías es {prob * 100:.2f}%')

elif exercise == 'Ejercicio 10':
    st.header('Ejercicio 10')
    st.write('**Enunciado:** El tiempo de reabastecimiento de cierto producto cumple con la distribución gamma con media de 40 y varianza de 400. Determina la probabilidad de que un pedido se envíe dentro de los 20 días posteriores a su solicitud y dentro de los primeros 60 días.')
    st.write('**Distribución de probabilidad utilizada:** Distribución gamma')
    st.write('**Datos:**')
    st.write('- Media (μ): 40')
    st.write('- Varianza (σ²): 400')
    
    mu = 40
    variance = 400
    k = mu**2 / variance
    theta = variance / mu
    
    prob_20_days = sp.gamma.cdf(20, a=k, scale=theta)
    prob_60_days = sp.gamma.cdf(60, a=k, scale=theta) - prob_20_days
    
    st.write('**Fórmulas:**')
    st.write('P(X ≤ 20) = CDF(20)')
    st.write('P(20 < X ≤ 60) = CDF(60) - CDF(20)')
    
    st.write('**Sustitución de los datos en la fórmula:**')
    st.write(f'P(X ≤ 20) = {prob_20_days:.4f}')
    st.write(f'P(20 < X ≤ 60) = {prob_60_days:.4f}')
    st.write(f'**Resultado:** La probabilidad de que un pedido se envíe dentro de los 20 días posteriores es {prob_20_days:.4f} y dentro de los primeros 60 días es {prob_60_days:.4f}')

    
elif exercise == 'Ejercicio 11':
    st.header('Ejercicio 11')
    st.write('**Enunciado:** El diámetro de unos ejes de acero sigue la distribución de Weibull con parámetros γ = 1.0 pulgadas, β = 2 y δ = 0.5. Encuentra la probabilidad de que un eje seleccionado al azar no exceda 1.5 pulgadas de diámetro.')
    st.write('**Distribución de probabilidad utilizada:** Distribución de Weibull')
    st.write('**Datos:**')
    st.write('- Parámetro de forma (β): 2')
    st.write('- Parámetro de escala (γ): 1.0')
    st.write('- Parámetro de ubicación (δ): 0.5')
    st.write('- Diámetro límite: 1.5')
    
    beta = 2
    gamma = 1.0
    delta = 0.5
    x = 1.5
    
    prob = sp.weibull_min.cdf(x, c=beta, scale=gamma, loc=delta)
    
    st.write('**Fórmula:**')
    st.write('P(X ≤ x) = 1 - e^(-(x - δ) / γ)^β')
    
    st.write('**Sustitución de los datos en la fórmula:**')
    st.write(f'P(X ≤ {x}) = 1 - e^(-(({x} - {delta}) / {gamma})^{beta})')
    st.write(f'**Resultado:** La probabilidad es {prob:.4f}')


elif exercise == 'Ejercicio 12':
    st.header('Ejercicio 12')
    st.write('**Enunciado:** La vida útil de una batería de celda seca se distribuye normalmente con media de 600 días y desviación estándar de 60 días.')
    st.write('a. ¿Qué fracción de estas baterías se esperaría que durara más de 680 días?')
    st.write('b. ¿Qué fracción de estas baterías se esperaría que fallara antes de 560 días?')
    st.write('**Distribución de probabilidad utilizada:** Distribución normal')
    st.write('**Datos:**')
    st.write('- Media (μ): 600 días')
    st.write('- Desviación estándar (σ): 60 días')
    
    mu = 600
    sigma = 60
    x1 = 680
    x2 = 560
    
    prob_more_680 = 1 - sp.norm.cdf(x1, loc=mu, scale=sigma)
    prob_less_560 = sp.norm.cdf(x2, loc=mu, scale=sigma)
    
    st.write('**Fórmulas:**')
    st.write('P(X > 680) = 1 - CDF(680)')
    st.write('P(X < 560) = CDF(560)')
    
    st.write('**Sustitución de los datos en la fórmula:**')
    st.write(f'P(X > {x1}) = 1 - CDF({x1})')
    st.write(f'P(X < {x2}) = CDF({x2})')
    st.write(f'**Resultado:** La fracción de baterías que durará más de {x1} días es {prob_more_680:.4f} y la fracción que fallará antes de {x2} días es {prob_less_560:.4f}')

