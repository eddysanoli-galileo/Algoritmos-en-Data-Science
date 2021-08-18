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
        String final que fue traducido a una función lambda. Útil para cerciorarse
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