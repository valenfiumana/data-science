# A partir de un estudio realizado por una asociación de conductores de autopista mostró que el 60% de los mismos utilizan el cinturón de seguridad correctamente.
# Si se selecciona una muestra de 10 conductores en una autopista.
# ¿Cuál es la probabilidad de que exactamente 7 de ellos lleven el cinturón de seguridad?

# Usando la fórmula binomial:
# X ~ Bi(n=10, p=0.6)
from math import factorial
import  math

# formula diplo
print(factorial(10)/(factorial(10-7)*factorial(7))*((0.6)**7)*((1-0.6)**(10-7))) # 0.21499084799999998

# formula proba
n = 10
p = 0.6
x = 7
print(math.comb(n, x) * p**x * (1-p)**(n-x)) # 0.21499084799999998

# Usando la función binomial de SciPy
from scipy.stats import binom
print(binom.pmf(7, 10, 0.6)) # 0.21499084799999976

# La proba es del 21%