import pandas as pd
import numpy as np

# ===========================================
# FUNCIONES: PARSING DE ECUACIONES
# ===========================================

def addCharBetweenMatch(string, regex, char):
    """
    Agrega un caracter entre las letras de un match dado por un regex. El match
    es luego re-concatenado con el resto del texto original. 

    Ejemplo: 
        Input : "Maria tiene 10 años"
        Match : "10"
        Char  : "0"
        Output: "Maria tiene 100 años"

    
    Parameters
    ----------
    regex : str
        Expresión regular definiendo el patrón al que se le hará "match"
    string : str
        String sobre el que se realizará el procesamiento
    char : str
        Caracter o caracteres a insertar entre cada letra o elemento del match

    Returns
    -------
    string: str
        String procesado con los caracteres adicionados
    """

    import re

    # Obtiene el primer match del regex en el string
    match = re.search(regex, string)

    while match:

        # Obtiene las diferentes partes del string
        # - strStart: Texto antes del match
        # - strMiddle: El match como tal
        # - strEnd: Texto luego del match
        strStart  = string[0:match.span()[0]]
        strMiddle = string[match.span()[0]:match.span()[1]]
        strEnd    = string[match.span()[1]:len(string)]

        # Se separa el "middle" en caracteres
        strMiddle = list(strMiddle)

        # Se agregan signos de multiplicación entre cada letra
        strMiddle = char.join(strMiddle)

        # Re-concatena cada parte del string
        string = strStart + strMiddle + strEnd

        # Vuelve a buscar matches luego del procesamiento
        match = re.search(regex, string)

    # Retorna el string una vez ya no encuentra más matches
    return string

def addParenthesis(string, regex):
    """
    Agrega paréntesis alrededor del segundo match de un regex.

    Ejemplo: 
        Input   : "Maria y Juan tienen 10 años"
        Match 1 : "Maria"
        Match 2 : "Juan"
        Output  : "Maria y (Juan) tienen 10 años"

    Parameters
    ----------
    regex : str
        Expresión regular definiendo el patrón al que se le hará "match". El regex
        debe de contener exactamente dos grupos nombrados.
    string : str
        String sobre el que se realizará el procesamiento


    Returns
    -------
    string: str
        String procesado con los paréntesis adicionados
    """

    import re

    # Se hace match para seleccionar el texto que se encerrará en paréntesis
    match = re.search(regex, string)

    # Si se encuentra un match, entonces se procesa. De lo contrario no
    if match:

        # Se separa el string en partes (Antes del match, el match y luego del match)
        strStart  = string[0:match.span()[0]]
        strMiddle = string[match.span()[0]:match.span()[1]]
        strEnd    = string[match.span()[1]:len(string)]

        # Se agregan paréntesis alrededor del segundo elemento
        strMiddle = match.groups()[0] + "(" + match.groups()[1] + ")"

        # Se concatenan nuevamente los strings
        string = strStart + strMiddle + strEnd

    else:
        pass

    return string


def parseEquation(formula):
    """
    Función que toma una ecuación escrita y la transforma en una función lambda

    Parameters
    ----------
    formula : str
        Ecuación escrita en notación tradicional 

    Returns
    -------
    f : lambda fun
        Función lambda creada a partir de la ecuación dada en la forma de un string
    lambda_str : str
        String final que fue traducido a una función lambda. �?til para cerciorarse
        que la función lambda tiene la estructura deseada. 
    """

    # Se importan los paquetes necesarios
    import ast
    import numpy as np
    import re

    # Se reemplazan los signos de "^" por el equivalente en Python ("**")
    formula = formula.replace("^", "**")

    # Se agregan paréntesis alrededor de los términos "elevados"
    expRegex = r"(?P<Ast>\*\*)(?P<Exp>[a-zA-Z0-9+\-\*\/]+)"
    formula = addParenthesis(formula, expRegex)

    # Se buscan sucesiones de letras y se separan por un "*"
    letterRegex = r"(?:[a-zA-Z])(?:[a-zA-Z]{1,})"
    formula = addCharBetweenMatch(formula, letterRegex, "*")

    # Se buscan números seguidos por letras y se separan por un "*"
    numberRegex = r"(?:[0-9])(?:[a-zA-Z])"
    formula = addCharBetweenMatch(formula, numberRegex, "*")

    # Se extraen todas las letras únicas presentes en la ecuación
    # Estas se consideran "variables" de la ecuación.
    variables = list(set(re.findall(r"[a-zA-Z]", formula)))

    # Se reemplaza la constante "e" por "np.exp"
    formula = formula.replace("e**", "np.exp")

    # Se elimina la constante "e" de las variables
    if "e" in variables:
        variables.remove("e")

    # Se construye el string de formula lambda
    # 1. "lambda"
    # 2. Todas las variables separadas por comas
    # 3. ":"
    # 4. La ecuación construida previamente
    lambda_str = "lambda " + ",".join(sorted(variables)) + " : " + formula

    # Se "parsea" la "lambda_str". El uso del parser "ast" evita la inyección de código malicioso
    code = ast.parse(lambda_str, mode="eval")

    # Se guarda el código evaluado en "f"
    f = eval(compile(code, "", mode="eval"))

    # Se devuelve la función lambda
    return f, lambda_str


# ===========================================
# FUNCIONES: DIFERENCIACIÓN
# ===========================================

def DiferFinitaCentrada1(f, x0, h):
    """
    Función que calcula la derivada de una función "f" en el punto "x0" utilizando
    dos miembros de la aproximación de Taylor.

    Parameters
    ----------
    f : fun
        Función unidimensional a derivar numéricamente.
    x0 : float
        Punto sobre el que se evaluará la derivada numérica.
    h : float
        Valor que determina la precisión con la que se realizará la aproximación
        de la derivada. Mientras más pequeño, mayor precisión.

    Returns
    -------
    df : float
        Aproximación numérica de la derivada de la función unidimensional.
    """

    f_sup = f(x0 + h)
    f_inf = f(x0 - h)
    df = (f_sup - f_inf) / (2*h)

    return df

def DiferFinitaCentrada2(f, x0, h):
    """
    Función que calcula la derivada de una función "f" en el punto "x0" utlizando
    cuatro miembros de la aproximación de Taylor.

    Parameters
    ----------
    f : fun
        Función unidimensional a derivar numéricamente.
    x0 : float
        Punto sobre el que se evaluará la derivada numérica.
    h : float
        Valor que determina la precisión con la que se realizará la aproximación
        de la derivada. Mientras más pequeño, mayor precisión.

    Returns
    -------
    df : float
        Aproximación numérica de la derivada de la función unidimensional.
    """

    f_sup  = f(x0 + h)
    f_inf  = f(x0 - h)
    f_sup2 = f(x0 + 2*h)
    f_inf2 = f(x0 - 2*h)
    df = (f_inf2 - 8*f_inf + 8*f_sup - f_sup2) / (12*h)

    return df

def DiferFinitaProgresiva(f, x0, h):
    """
    Función que calcula la derivada de una función "f" en el punto "x0" utlizando
    una diferencia finita progresiva.

    Parameters
    ----------
    f : fun
        Función unidimensional a derivar numéricamente.
    x0 : float
        Punto sobre el que se evaluará la derivada numérica.
    h : float
        Valor que determina la precisión con la que se realizará la aproximación
        de la derivada. Mientras más pequeño, mayor precisión.

    Returns
    -------
    df : float
        Aproximación numérica de la derivada de la función unidimensional.
    """

    f_x  = f(x0)
    f_h  = f(x0 + h)
    f_2h = f(x0 + 2*h)
    df = (-3*f_x + 4*f_h - f_2h) / (2*h)

    return df


# ===========================================
# FUNCIONES: CÁLCULO DE SOLUCIONES
# ===========================================

def SolverNewton(F, x0, k_max, epsilon):
    """ 
    Función para obtener los puntos estacionarios de una función F(x) = 0 a través del
    método de Newton-Raphson.

    ...

    Parameters
    ----------
    f : function
        Función unidimensional diferenciable
    x0 : float
        Solución inicial a la ecuación.
    k_max : int
        Número máximo de iteraciones para el algoritmo
    epsilon : float
        Tolerancia de error empleada para detener el algoritmo en caso alcance una
        cierta precisión deseada.


    Returns
    -------
    xk : float
        Aproximación a la raíz de la ecuación F(x) = 0
    table : pandas dataframe
        Dataframe conteniendo un resumen de la aproximación xk y el error en cada
        iteración "k".

    """
    # Se inicializan las iteraciones
    k = 0 

    # Se inicializa la aproximación de la raíz de la ecuación
    xk = x0

    # Se inicializa el dataframe que almacenará los datos de cada iteración
    table = pd.DataFrame(columns = ["Iter", "Xk", "Error"])

    while (k < k_max) and (abs(F(xk)) > epsilon):

        # Se calcula la derivada de la función en el punto xk
        dF = DiferFinitaCentrada2(F, xk, 0.00001)

        # Para evitar divisiones entre 0, si la derivada es igual a cero se 
        # reemplaza por un valor muy muy pequeño
        if dF == 0:
            dF = 0.000000001

        # Se actualiza xk
        xk = xk - (F(xk) / dF)

        # Se incrementan las iteraciones
        k += 1

        # Se agrega la información actual al dataframe
        table = table.append({"Iter": k, "Xk": xk, "Error": np.abs(F(xk))}, ignore_index=True)

    return xk, table


def SolverBiseccion(F, lim_inf, lim_sup, k_max, epsilon):
    """ 
    Función para obtener los puntos estacionarios de una función F(x) = 0 a través del 
    método de Bisección. Si, dadas las condiciones iniciales proporcionadas, se determina
    que el problema divergerá, se retorna una excepción.

    ...

    Parameters
    ----------
    F : function
        Función unidimensional continua en un "intervalo" y que cambia de signo dentro
        de dicho intervalo. 
    intervalo : list
        Intervalo en el que la función es continua y cambia de signo. Consiste de una 
        lista con la siguiente forma [límite_inf, límite_sup] o (a, b)
    k_max : int
        Número máximo de iteraciones para el algoritmo
    epsilon : float
        Tolerancia de error empleada para detener el algoritmo en caso alcance una
        cierta precisión deseada.


    Returns
    -------
    xk : float
        Aproximación a una raíz de la ecuación F(x) = 0 en el intervalo (a, b)
    df : pandas dataframe
        Dataframe conteniendo un resumen de la aproximación xk y el error en cada
        iteración "k".

    """
    

    # Se obtiene el límite superior e inferior del intervalo
    a, b = lim_inf, lim_sup

    # Si la multiplicación de las dos evaluaciones es mayor o igual a 0 entonces
    # el método fallará desde un inicio.
    if F(a)*F(b) >= 0:
        raise Exception("El método de bisección divergerá al utilizar este intervalo")

    # Se inicializan las iteraciones
    k = 0 

    # Se inicializa la aproximación de la raíz de la ecuación
    xk = (a + b) / 2

    # Se inicializa el dataframe que almacenará los datos de cada iteración
    df = pd.DataFrame(columns = ["Iter", "Xk", "Error"])

    while (k < k_max) and (abs(F(xk)) > epsilon):

        if (F(a)*F(xk)) < 0:
            b = xk
        else:
            a = xk

        # Se incrementan las iteraciones y 
        k += 1
        xk = (a + b) / 2

        # Se agrega la información actual al dataframe
        df = df.append({"Iter": k, "Xk": xk, "Error": np.abs(F(xk))}, ignore_index=True)

    return xk, df

# ===========================================
# FUNCIONES: DESCENSO DE GRADIENTE
# ===========================================

def arr2str(array):
    """
    Función que convierte un array o lista en un único string, convirtiendo cada elemento
    en un string y luego concatenándolo al string. 

    Parameters
    ----------
    array : list or np.array
        Vector o lista de elementos que se desean combinar para crear un string.

    Returns
    -------
    arr_str: str
        Conversión de la lista o vector dado a string. 

    """

    # Se redondea el array para que tenga 4 decimales
    array = np.around(array, 4)

    # Se convierte el array en un único string
    arr_str = ' '.join(map(str, array))

    return arr_str

def gradientDescent(f, df, x0, epsilon = 1E-6, max_iter = 100, step_size_type = "Constante", step_size_val = 0.001, alpha_params=None): 
    """ 
    Función para ejecutar el algoritmo de descenso de gradiente para una función dada.
    Dadas las limitaciones del código empleado, no se debe proveer la función a optimizar 
    como tal, sino la derivada de la misma. 

    ...

    Parameters
    ----------
    f: lambda function
        Función a optimizar. Cada vector alimentado a la función debe tener tantas
        filas como muestras y tantas columnas como dimensiones.
    df : lambda function
        Derivada o gradiente de la función a optimizar. Debe de contar con la misma 
        cantidad de variables que `x0`. 
    x0 : array
        Vector que contiene el punto inicial desde el que el algoritmo de descenso 
        de gradiente inicia la búsqueda del minimizador.
    epsilon : float
        Tolerancia de error empleada para detener el algoritmo en caso alcance una
        cierta precisión deseada.
    max_iter : int
        Número máximo de iteraciones que puede llegar a ejecutar el algoritmo durante
        su búsqueda.
    step_size_tipe : str
        Tipo de step size a utilizar durante las iteraciones. Existen 3 tipos: 'Exacto',
        'Constante' y 'Variable'. Para el 'Exacto', el algoritmo resuelve un sub-problema
        de optimización para determinar el tamaño de step-size óptimo. Para el 'Constante',
        el usuario debe especificar un valor para el step-size a través del parámetro
        `step_size_val`. Para el 'Variable', el step-size sigue la secuencia de decrecimiento
        1/k, donde `k` consiste del número de iteración (step-size disminuye con el avance del
        algoritmo). Default: 'Constante'.
    step_size_val : float, optional
        Valor para el step-size cuando se selecciona el tipo de step-size 'Constante'
    step_size_params : dict, optional
        Diccionario conteniendo los valores para todas las variables que deben utilizarse 
        durante la sub-optimización ejecutada al seleccionar un tipo de step size 'Exacto'.


    Returns
    -------
    table : pandas dataframe
        Dataframe conteniendo un resumen de la aproximación `xk`, el gradiente `pk` y el error 
        `||df||` en cada iteración `k`.

    """

    # Importación de paquetes
    from numpy.linalg import norm

    # Se coloca x0 como primer valor de xk
    xk = x0.copy()

    # Iteración actual
    k = 0                           

    if len(xk) > 5:

        # Se inicializa el dataframe que almacenará los datos de cada iteración
        table = pd.DataFrame(columns = ["k", "||df||", "f"])

        # Se agregan los datos de la primera fila
        table = table.append({"k": k, "||df||": norm(df(xk)), "f": f(xk)}, ignore_index=True)

    else: 
        # Se inicializa el dataframe que almacenará los datos de cada iteración
        table = pd.DataFrame(columns = ["k", "Xk", "Pk", "||df||", "f"])

        # Se agregan los datos de la primera fila
        table = table.append({"k": k, "Xk": arr2str(xk), "Pk": arr2str(df(xk)), "||df||": norm(df(xk)), "f": f(xk)}, ignore_index=True)

    # Método de GD:
    # - Determinar la dirección pk
    # - Determinar el step size ak
    # - Iterar sobre la aproximación inicial utilizando x_k+1 = xk + ak*pk
    while (norm(df(xk)) >= epsilon) and (k < max_iter):

        # Se incrementa el número de iteraciones
        k += 1

        # Dirección: grad(f)
        grad = df(xk)

        # Step size:
        # - Exacto: Obtenido al solucionar un problema de optimización
        # - Constante: Valor constante para toda iteración
        # - Variable: Valor dado según sucesión 1/k
        if step_size_type == "Exacto":
            
            # Ver pág 23 de Lecture 6. Solo para la ecuación cuadrática
            alpha = (grad.T @ grad) / (grad.T @ alpha_params["Q"] @ grad)

        elif step_size_type == "Constante":
            alpha = step_size_val

        elif step_size_type == "Variable":
            # Sucesión decreciente
            alpha = 1/k

        # Iteración: 
        xk = xk - alpha * grad


        if len(xk) > 5:
            table = table.append({"k": k, "||df||": norm(grad), "f": f(xk)}, ignore_index=True)

        else:
            # Se agrega la información actual al dataframe
            # Los datos deben ser convertidos en strings para evitar problemas
            # posteriores cuando R intenta mostrar los datos. Si no se hace esto
            # se queja de que se le están pasando más datos de los que se deberían,
            # porque parece tomar las listas dentro del dataframe como columnas adicionales
            table = table.append({"k": k, "Xk": arr2str(xk), "Pk": arr2str(grad), "||df||": norm(grad), "f": f(xk)}, ignore_index=True)

    return table


def SolverQP(Q, c, x0, epsilon, max_iter, step_size_type, step_size_val=0.0001):
    """ 
    Función para ejecutar el algoritmo de descenso de gradiente en el denominado 
    "problema cuadrático" (QP) con la forma: 0.5 x^T Q x + c^T x.

    ...

    Parameters
    ----------
    Q : array
        Matriz cuadrada con tantas filas y columnas, como valores existen en x0. Mientras
        esta matriz sea positiva definida, el problema de QP será de carácter convexo.
    c : array
        Vector con la misma forma que x0. Estos consisten de los componentes lineales
        del problema cuadrático.
    x0 : array
        Vector que contiene el punto inicial desde el que el algoritmo de descenso 
        de gradiente inicia la búsqueda del mínimo.
    epsilon : float
        Tolerancia de error empleada para detener el algoritmo en caso alcance una
        cierta precisión deseada.
    max_iter : int
        Número máximo de iteraciones que puede llegar a ejecutar el algoritmo durante
        su búsqueda.
    step_size_tipe : str
        Tipo de step size a utilizar durante las iteraciones. Existen 3 tipos: 'Exacto',
        'Constante' y 'Variable'. Para el 'Exacto', el algoritmo resuelve un sub-problema
        de optimización para determinar el tamaño de step-size óptimo. Para el 'Constante',
        el usuario debe especificar un valor para el step-size a través del parámetro
        `step_size_val`. Para el 'Variable', el step-size sigue la secuencia de decrecimiento
        1/k, donde `k` consiste del número de iteración (step-size disminuye con el avance del
        algoritmo). Default: 'Constante'.
    step_size_val : float, optional
        Valor para el step-size cuando se selecciona el tipo de step-size 'Constante'


    Returns
    -------
    table : pandas dataframe
        Dataframe conteniendo un resumen de la aproximación `xk`, el gradiente `pk` y el error 
        `||df||` en cada iteración `k`.

    """

    # Se convierten todos los inputs vectoriales a arrays de numpy
    Q = np.array(Q)
    c = np.array(c)
    x0 = np.array(x0)

    # Función cuadrática
    f = lambda x: 0.5 * x.T @ Q @ x + c.T @ x

    # Gradiente de función cuadrática
    # A partir de Python 3.5 la multiplicación matricial se puede hacer con "@"
    df = lambda x: Q @ x + c

    # Se ejecuta el algoritmo y se muestra la tabla
    table = gradientDescent(f, df, x0, epsilon=epsilon, max_iter=max_iter, step_size_type=step_size_type, step_size_val=step_size_val, alpha_params={"Q": Q})

    return table


def SolverRosenbrock(x0, epsilon, max_iter, step_size_val=0.05):
    """ 
    Función para ejecutar el algoritmo de descenso de gradiente en la función
    de Rosenbrock.

    .. math:: f\left(x_{1}, x_{2}\right)=100\left(x_{2}-x_{1}^{2}\right)^{2}+\left(1-x_{1}\right)^{2}

    Parameters
    ----------
    x0 : array
        Vector que contiene el punto inicial desde el que el algoritmo de descenso 
        de gradiente inicia la búsqueda del mínimo.
    epsilon : float
        Tolerancia de error empleada para detener el algoritmo en caso alcance una
        cierta precisión deseada.
    max_iter : int
        Número máximo de iteraciones que puede llegar a ejecutar el algoritmo durante
        su búsqueda.
    step_size_val : float, optional
        Valor para el step-size cuando se selecciona el tipo de step-size 'Constante'


    Returns
    -------
    table : pandas dataframe
        Dataframe conteniendo un resumen de la aproximación `xk`, el gradiente `pk` y el error 
        `||df||` en cada iteración `k`.

    """

    # Función Rosenbrock
    f = lambda x : 100*(x[1,0] - x[0,0]**2)**2 + (1-x[0,0])**2

    # Gradiente de función Rosenbrock
    # A partir de Python 3.5 la multiplicación matricial se puede hacer con "@"
    df = lambda x: np.array([[100*( 4*x[0,0]**3 - 4*x[0,0]*x[1,0]) + 2*x[0,0] - 2],
                             [100*(-2*x[0,0]**2 + 2*x[1,0])                     ]])

    # Se ejecuta el algoritmo y se muestra la tabla
    table = gradientDescent(f, df, x0, epsilon=epsilon, max_iter=max_iter, step_size_type="Constante", step_size_val=step_size_val)

    return table

# ===========================================
# FUNCIONES: VARIANTES DE DESCENSO DE GRADIENTE
# ===========================================

def solCerrada(n, d):
    """
    Función que devuelve la solución exacta a una regresión lineal con la forma
    Ax = b. Advertencia: No utilizar dimensionalidades demasiado altas.

    Parameters
    ----------
    n : int
        Número de muestras a incluir en las matrices y vectores del problema.
    d : int
        Número de dimensiones a incluir en las matrices y vectores del problema.

    Returns
    -------
    x : np.array
        Vector con la solución cerrada del problema de regresión lineal.
    """

    # Matriz A (1000 x 100)
    # Valores extraídos independientemente de una distribución Gaussiana de "d" dimensiones
    A = np.random.normal(0, 1, size=(n,d))      

    # Vector de coeficientes "verdadero" (100 x 1)
    # Valores extraídos de una distribución Gaussiana de "d" dimensiones
    X_true = np.random.normal(0, 1, size=(d,1))

    # Vector de coeficientes "b"  (1000, 1)
    # Se asigna a cada observación a_i la etiqueta "(x*)T a_i" más un componente de ruido
    b = A.dot(X_true) + np.random.normal(0, 0.5, size=(n,1))

    # Solución cerrada
    x_star = np.linalg.inv(A.T @ A) @ A.T @ b

    return arr2str(x_star)


def stochasticGradientDescent(f, df, x0, epsilon = 1E-6, max_iter = 100, step_size = 0.001): 
    """ 
    Función para ejecutar el algoritmo de descenso de gradiente estocástico para una función 
    dada. Esto implica que las muestras alimentadas para el entrenamiento son "mezcladas" de
    forma aleatoria y a continuación son utilizadas una a una para actualizar el valor de xk.
    ...

    Parameters
    ----------
    f: lambda function
        Función a optimizar. Cada vector alimentado a la función debe tener tantas
        filas como muestras y tantas columnas como dimensiones.
    df : lambda function
        Derivada o gradiente de la función a optimizar. Debe de contar con la misma 
        cantidad de variables que `x0`. 
    x0 : array
        Vector que contiene el punto inicial desde el que el algoritmo de descenso 
        de gradiente inicia la búsqueda del minimizador.
    epsilon : float
        Tolerancia de error empleada para detener el algoritmo en caso alcance una
        cierta precisión deseada.
    max_iter : int
        Número máximo de iteraciones que puede llegar a ejecutar el algoritmo durante
        su búsqueda.
    step_size : float, optional
        Valor para el step-size o learning rate.

    Returns
    -------
    table : pandas dataframe
        Dataframe conteniendo un resumen de la aproximación `xk`, el gradiente `pk` y el error 
        `||df||` en cada iteración `k`.
    """

    # Paquetes necesarios
    from numpy.linalg import norm

    # Se coloca x0 como primer valor de xk
    xk = x0.copy()

    # Iteración actual e índice actual
    k = 0                           
    idx = 0

    # Se inicializa el dataframe que almacenará los datos de cada iteración
    table = pd.DataFrame(columns = ["k", "||df||", "f"])

    # Se agregan los datos de la primera fila
    table = table.append({"k": 0, "||df||": norm(df(xk,idx)), "f": f(xk)}, ignore_index=True)

    # Se crea un vector con números de 1 hasta "n"
    # Las filas del vector se "shufflean"
    idxs = np.arange(0, len(xk), 1)
    idxs = np.random.permutation(idxs)

    # Método de GD:
    # - Determinar la dirección pk
    # - Determinar el step size ak
    # - Iterar sobre la aproximación inicial utilizando x_k+1 = xk + ak*pk
    while (norm(df(xk, idx)) >= epsilon) and (k < max_iter):

        # Se toman las muestras una a una y se utilizan
        # para actualizar al vector X
        for idx in idxs:

            # Dirección: grad(f)
            grad = df(xk, idx)

            # Iteración:
            xk = xk - step_size * grad

            # Se incrementa el número de iteraciones
            k += 1

            # Se agrega la información actual al dataframe
            table = table.append({"k": k, "||df||": norm(grad), "f": f(xk)}, ignore_index=True)

            # Si se alcanza la precisión deseada, se interrumpe el "for"
            if not ((norm(df(xk,idx)) >= epsilon) and (k < max_iter)):
                break

    return table


def minibatchGradientDescent(f, df, x0, batch_size=25, epsilon = 1e-6, max_iter = 100, learning_rate = 0.001, params={"A": 0, "b": 0}): 
    """ 
    Función para ejecutar el algoritmo de descenso de gradiente con mini-batch para una función 
    dada. Esto implica que las muestras alimentadas para el entrenamiento son "mezcladas" de
    forma aleatoria y a continuación son utilizadas en bloques denominados "batches" para 
    actualizar el valor de xk.
    ...

    Parameters
    ----------
    f: lambda function
        Función a optimizar. Cada vector alimentado a la función debe tener tantas
        filas como muestras y tantas columnas como dimensiones.
    df : lambda function
        Derivada o gradiente de la función a optimizar. Debe de contar con la misma 
        cantidad de variables que `x0`. 
    batch_size : int
        Número de muestras que son procesadas simultáneamente por el algoritmo durante
        el entrenamiento. 
    x0 : array
        Vector que contiene el punto inicial desde el que el algoritmo de descenso 
        de gradiente inicia la búsqueda del minimizador.
    epsilon : float
        Tolerancia de error empleada para detener el algoritmo en caso alcance una
        cierta precisión deseada.
    max_iter : int
        Número máximo de iteraciones que puede llegar a ejecutar el algoritmo durante
        su búsqueda.
    learning_rate : float, optional
        Valor para el step-size o learning rate.
    params : dict 
        Diccionario conteniendo los parámetros de la función "f" y su derivada "df". 

    Returns
    -------
    table : pandas dataframe
        Dataframe conteniendo un resumen de la aproximación `xk`, el gradiente `pk` y el error 
        `||df||` en cada iteración `k`.
    """

    # Paquetes necesarios
    from numpy.linalg import norm

    # Se coloca x0 como primer valor de xk
    xk = x0.copy()

    # Iteración actual e índice actual
    k = 0                           

    # =============
    # PREPARACIÓN BATCHES
    # =============

    # Se crea un vector con números de 1 hasta "n"
    # Las filas del vector se "shufflean"
    idxs = np.arange(0, len(xk), 1)
    idxs = np.random.permutation(idxs)

    # Los índices "shuffleados" se usan para "mezclar" las filas de "A" y "b" 
    A = params["A"].take(idxs, 0)
    b = params["b"].take(idxs, 0)

    # Se extrae el número de batches (en forma de int)
    num_batches = xk.shape[0] // batch_size

    # =============
    # TABLA
    # =============

    # Se inicializa el dataframe que almacenará los datos de cada iteración
    table = pd.DataFrame(columns = ["k", "||df||", "f"])

    # Se agregan los datos de la primera fila
    table = table.append({"k": 0, "||df||": norm(df(xk, A, b)), "f": f(xk)}, ignore_index=True)

    # Método de GD:
    # - Determinar la dirección pk
    # - Determinar el step size ak
    # - Iterar sobre la aproximación inicial utilizando x_k+1 = xk + ak*pk
    while (norm(df(xk, A, b)) >= epsilon) and (k < max_iter):

        # Para cada batch
        for batch in range(num_batches):

            # Inicio de batch: Número de batch por el batch size
            strt = batch*batch_size

            # Fin de batch: Inicio + batch size - 1
            if batch == num_batches-1:
                end = xk.shape[0]
            else:
                end = strt + batch_size - 1

            # Se selecciona el "batch" de datos a utilizar de A y b
            A_sel = A[strt:end, :]
            b_sel = b[strt:end, :]

            # Dirección: grad(f)
            grad = df(xk, A_sel, b_sel)

            # Iteración:
            xk = xk - learning_rate * grad

            # Se incrementa el número de iteraciones
            k += 1

            # Se agrega la información actual al dataframe
            table = table.append({"k": k, "||df||": norm(grad), "f": f(xk)}, ignore_index=True)

            # Si se alcanza la precisión deseada, se interrumpe el "for"
            if not ((norm(df(xk, A, b)) >= epsilon) and (k < max_iter)):
                break

    return table


def SolverLinReg(GDType="batch", epsilon=1e-8, max_iter=1000, step_size=0.01, batch_size=25):
    """
    Método empleado para resolver la regresión lineal con la forma Ax = b, empleando
    diferentes variantes de descenso de gradiente: Batch, estocástico y mini batch. Los
    valores de las matrices A y b son tomados aleatoriamente de una distribución normal
    con media 0 y desviación estándar 1. Se generan 1000 muestras con 100 dimensiones, 
    lo que implica una matriz A de (1000x100) y una b de (1000,1).

    Parameters
    ----------
    GDType : str, optional
        Tipo de descenso de gradiente a utilizar
    epsilon : float
        Tolerancia de error empleada para detener el algoritmo en caso alcance una
        cierta precisión deseada.
    max_iter : int
        Número máximo de iteraciones que puede llegar a ejecutar el algoritmo durante
        su búsqueda.
    step_size : float, optional
        Valor para el step-size o learning rate.
    batch_size : int
        Número de muestras que son procesadas simultáneamente por el algoritmo durante
        el entrenamiento. 

    Returns
    -------
    table : pandas dataframe
        Dataframe conteniendo los resultados del algoritmo seleccionado. 
    """

    # -----------------
    # GENERACIÓN DE DATOS
    # -----------------

    # Columnas del dataset
    d = 100

    # Observaciones del dataset
    n = 1000

    # Matriz A (1000 x 100)
    # Valores extraídos independientemente de una distribución Gaussiana de "d" dimensiones
    A = np.random.normal(0, 1, size=(n,d))      

    # Vector de coeficientes "verdadero" (100 x 1)
    # Valores extraídos de una distribución Gaussiana de "d" dimensiones
    X_true = np.random.normal(0, 1, size=(d,1))

    # Vector de coeficientes "b"  (1000, 1)
    # Se asigna a cada observación a_i la etiqueta "(x*)T a_i" más un componente de ruido
    b = A.dot(X_true) + np.random.normal(0, 0.5, size=(n,1))

    # -----------------
    # FUNCIÓN OBJETIVO 
    # -----------------

    # Función objetivo
    f = lambda x : np.sum((A @ x - b)**2, axis=0, keepdims=True).item()

    # Vector inicial
    x0 = np.zeros_like(X_true)

    # -----------------
    # ENTRENAMIENTO
    # -----------------

    if GDType == "batch":

        # Derivada de la función objetivo
        df = lambda x : 2 * A.T @ (A @ x - b)

        # Entrenamiento
        tabla = gradientDescent(f, df, x0, step_size_type="Constante", step_size_val=step_size, max_iter=max_iter)

    if GDType == "stochastic":

        # Se cambia la derivada que sea posible "indexarla" fila a fila
        df = lambda x, idx: 2*np.reshape(A[idx,:], (-1,1)) @ (x.T @ np.reshape(A[idx,:], (-1,1)) - b[idx,0])

        # Entrenamiento
        tabla = stochasticGradientDescent(f, df, x0, epsilon=epsilon, step_size=step_size, max_iter=max_iter)

    if GDType == "mini-batch":

        # A la derivada se le pasan A y b como parámetros adicionales
        df = lambda x, A, b : 2 * A.T @ (A @ x - b)

        # Entrenamiento
        tabla = minibatchGradientDescent(f, df, x0, batch_size=batch_size, learning_rate=step_size, max_iter=max_iter, params={"A": A, "b": b})

    # Se retorna la tabla generada
    return tabla


# ===========================================
# FUNCIONES: BACKTRACKING LINE SEARCH
# ===========================================

def gradientDescentBLS(x0, epsilon = 1e-6, max_iter = 100):
    """ 
    Función para ejecutar el algoritmo de descenso de gradiente, con selección de
    step-size por medio del método de backtracking line search, el cual consiste en
    iterar sobre el valor del step size hasta cumplir con las condiciones de armijo.
    ...

    Parameters
    ----------
    x0 : array
        Vector que contiene el punto inicial desde el que el algoritmo de descenso 
        de gradiente inicia la búsqueda del minimizador.
    epsilon : float
        Tolerancia de error empleada para detener el algoritmo en caso alcance una
        cierta precisión deseada.
    max_iter : int
        Número máximo de iteraciones que puede llegar a ejecutar el algoritmo durante
        su búsqueda.

    Returns
    -------
    table : pandas dataframe
        Dataframe conteniendo un resumen de la aproximación `xk`, el valor de la función 
        `f` y el error `||df||` en cada iteración `k`.
    """

    # Función de Rosenbrock
    f = lambda x : 100*(x[1,0] - x[0,0]**2)**2 + (1-x[0,0])**2

    # Gradiente de función de Rosenbrock
    df = lambda x: np.array([[-400*(-x[0,0]**3 + x[0,0]*x[1,0]) - 2*(1-x[0,0])],
                             [ 200*(-x[0,0]**2 + x[1,0])                      ]])

    # Paquetes necesarios
    from numpy.linalg import norm

    # Se coloca x0 como primer valor de xk
    xk = x0.copy()

    # Iteración actual
    k = 0                           

    # Se inicializa el dataframe que almacenará los datos de cada iteración
    table = pd.DataFrame(columns = ["k", "Xk", "||df||", "f"])

    # Se agregan los datos de la primera fila
    table = table.append({"k": 0, "Xk": arr2str(xk), "||df||": norm(df(xk)), "f": f(xk)}, ignore_index=True)

    # Método de GD:
    # - Determinar la dirección pk
    # - Determinar el step size ak
    # - Iterar sobre la aproximación inicial utilizando x_k+1 = xk + ak*pk
    while (norm(df(xk)) >= epsilon) and (k < max_iter):

        # Se incrementa el número de iteraciones
        k += 1

        # Dirección: grad(f)
        pk = df(xk)

        # =====================
        # Backtracking Line Search
        
        # Se inicializan los parámetros
        alpha_bar = 1
        rho = 0.5
        c = 1e-4

        # Se asigna alpha_bar a alpha
        alpha = alpha_bar
        
        # Se itera hasta que la condición esa falsa
        # (Se debe cumplir la condición de Armijo)
        while f(xk + alpha*pk) > (f(xk) + c*alpha * df(xk).T @ pk):
            alpha = rho * alpha
        
        # =====================

        # Iteración:
        xk = xk - alpha * pk

        # Se agrega la información actual al dataframe
        table = table.append({"k": k, "Xk": arr2str(xk), "||df||": norm(pk), "f": f(xk)}, ignore_index=True)

    return table


def NewtonBLS(x0, epsilon = 1e-6, max_iter = 100, alpha_override = None):
    """ 
    Función para ejecutar el método de Newton, con selección de step-size por 
    medio del método de backtracking line search, el cual consiste en iterar 
    sobre el valor del step size hasta cumplir con las condiciones de armijo.
    ...

    Parameters
    ----------
    x0 : array
        Vector que contiene el punto inicial desde el que el algoritmo de descenso 
        de gradiente inicia la búsqueda del minimizador.
    epsilon : float
        Tolerancia de error empleada para detener el algoritmo en caso alcance una
        cierta precisión deseada.
    max_iter : int
        Número máximo de iteraciones que puede llegar a ejecutar el algoritmo durante
        su búsqueda.
    alpha_override : float or None
        Si se especifica un valor para este parámetro, el mismo se utiliza como sustitución
        para el valor de step size calculado por medio de backtracking line search.

    Returns
    -------
    table : pandas dataframe
        Dataframe conteniendo un resumen de la aproximación `xk`, el valor de la función 
        `f` y el error `||df||` en cada iteración `k`.
    """

    # Función de Rosenbrock
    f = lambda x : 100*(x[1,0] - x[0,0]**2)**2 + (1-x[0,0])**2

    # Gradiente de función de Rosenbrock
    df = lambda x: np.array([[-400*(-x[0,0]**3 + x[0,0]*x[1,0]) - 2*(1-x[0,0])],
                             [ 200*(-x[0,0]**2 + x[1,0])                      ]])

    # Hessiano de la función de Rosenbrock
    ddf = lambda x: np.array([[-400*(-3*x[0,0]**2 + x[1,0]) + 2, -400*x[0,0]],
                              [                     -400*x[0,0],         200]])

    # Paquetes necesarios
    from numpy.linalg import norm, inv

    # Se coloca x0 como primer valor de xk
    xk = np.array(x0)

    # Iteración actual
    k = 0  

    # Primera predicción de la dirección
    pk = -inv(ddf(xk)) @ df(xk)                         

    # Se inicializa el dataframe que almacenará los datos de cada iteración
    table = pd.DataFrame(columns = ["k", "Xk", "Pk", "||df||"])

    # Se agregan los datos de la primera fila
    table = table.append({"k": 0, "Xk": arr2str(xk), "Pk": arr2str(pk), "||df||": norm(df(xk))}, ignore_index=True)

    # Método de GD:
    # - Determinar la dirección pk
    # - Determinar el step size ak
    # - Iterar sobre la aproximación inicial utilizando x_k+1 = xk + ak*pk
    while (norm(df(xk)) >= epsilon) and (k < max_iter):

        # Se incrementa el número de iteraciones
        k += 1

        # Dirección: -[ddf(x)]^-1 df(x)
        pk = -inv(ddf(xk)) @ df(xk)

        # =====================
        # Backtracking Line Search
        
        # Se inicializan los parámetros
        alpha_bar = 1
        rho = 0.5
        c = 1e-4

        # Se asigna alpha_bar a alpha
        alpha = alpha_bar
        
        # Se itera hasta que la condición esa falsa
        # (Se debe cumplir la condición de Armijo)
        while f(xk + alpha*pk) > (f(xk) - c*alpha * df(xk).T @ pk):
            alpha = rho * alpha
        
        # =====================

        if alpha_override:
            alpha = alpha_override

        # Iteración:
        xk = xk + alpha * pk

        # Se agrega la información actual al dataframe
        table = table.append({"k": k, "Xk": arr2str(xk), "Pk": arr2str(pk), "||df||": norm(df(xk))}, ignore_index=True)

    return table
