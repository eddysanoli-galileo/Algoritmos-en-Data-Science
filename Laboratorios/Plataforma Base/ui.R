library(shiny)
library(shinydashboard)

# Define UI for application that draws a histogram
dashboardPage(
    dashboardHeader(title = "Algoritmos en DS"),
    dashboardSidebar(
        sidebarMenu(
            
            # Calcular los ceros de una ecuación
            menuItem("Ceros", tabName = "Ceros",
                     menuSubItem("Bisección", tabName = "Biseccion"),
                     menuSubItem("Newton-Raphson", tabName = "Newton")
                     ),
            
            # Derivar numéricamente una función
            menuItem("Derivación", tabName = "Derivacion",
                     menuSubItem("Diferencia Finita Centrada 1", tabName = "DiferFinCentr1"),
                     menuSubItem("Diferencia Finita Centrada 2", tabName = "DiferFinCentr2"),
                     menuSubItem("Diferencia Finita Progresiva", tabName = "DiferFinProg")
                     ),
            
            # Descenso de Gradiente en Rosenbrock y QP
            menuItem("Gradient Descent", tabName = "GD",
                     menuSubItem("Problema Cuadrático (QP)", tabName = "QP"),
                     menuSubItem("Función de Rosenbrock", tabName = "Rosenbrock")
                     ),
            
            
            # Variantes de descenso de gradiente
            menuItem("Gradient Descent Variants", tabName = "var_GD",
                     menuSubItem("Solución Cerrada", tabName = "SC"),
                     menuSubItem("Batch Gradient Descent", tabName = "BGD"),
                     menuSubItem("Stochastic Gradient Descent", tabName = "SGD"),
                     menuSubItem("Mini-batch Gradient Descent", tabName = "MBGD")
                     ),
            
            # Backtracking Line Search
            menuItem("Backtracking Line Search", tabName = "BLS",
                     menuSubItem("Gradient Descent", tabName = "GDBLS"),
                     menuSubItem("Método de Newton", tabName = "NBLS")
                     )
            
        )
    ),
    
    dashboardBody(
        tabItems(
            
            # ====================
            # CEROS
            # ====================
            
            # Ceros - Método de Newton 
            tabItem("Newton",
                    h1("Método de Newton-Raphson"),
                    h3("Rutina para obtener los puntos estacionarios de una ecuación con la forma F(x) = 0 a través del método de Newton-Raphson."),
                    box(textInput("ecuacionNew", "Ecuación a Evaluar"),
                        textInput("x0New", "Solucion Inicial de Ecuación (X0)"),
                        textInput("maxIterNew", "Numero Máximo de Iteraciones (k_max)"),
                        textInput("epsilonNew", "Precision a Alcanzar (epsilon)")),
                    actionButton("nwtSolver", "Resolver"),
                    tableOutput("salidaNewton")),
            
            # Ceros - Método de Bisección
            tabItem("Biseccion",
                    h1("Método de Bisección"),
                    h3("Función para obtener los puntos estacionarios de una función F(x) = 0 a través del método de Bisección"),
                    box(textInput("ecuacionBis", "Ecuación a Evaluar"),
                        textInput("lim_infBis", "Límite Inferior de Intervalo de Búsqueda (a)"),
                        textInput("lim_supBis", "Límite Superior de Intervalo de Búsqueda (b)"),
                        textInput("maxIterBis", "Numero Máximo de Iteraciones (k_max)"),
                        textInput("epsilonBis", "Precision a Alcanzar (epsilon)")),
                    actionButton("bisSolver", "Resolver"),
                    tableOutput("salidaBiseccion")),
            
            # ====================
            # DIFERENCIACIÓN
            # ====================
            
            # Diferenciación - Diferencia Finita Centrada 1
            tabItem("DiferFinCentr1",
                    h1("Diferencia Finita Centrada 1"),
                    h3("Función que calcula la derivada de una función 'f' en el punto 'x0' utilizando dos miembros de la aproximación de Taylor."),
                    box(textInput("ecuacion_Dif1", "Ecuación a Evaluar"),
                        textInput("x0_Dif1", "Punto para el que se Calculará la Derivada Numérica (x0)"),
                        textInput("h_Dif1", "Precisión a Utilizar (h)")),
                    actionButton("DiferFinCentr1_Solver", "Calcular Derivada"),
                    textOutput("DiferFinCentr1_Out")),
            
            # Diferenciación - Diferencia Finita Centrada 1
            tabItem("DiferFinCentr2",
                    h1("Diferencia Finita Centrada 2"),
                    h3("Función que calcula la derivada de una función 'f' en el punto 'x0' utilizando cuatro miembros de la aproximación de Taylor."),
                    box(textInput("ecuacion_Dif2", "Ecuación a Evaluar"),
                        textInput("x0_Dif2", "Punto para el que se Calculará la Derivada Numérica (x0)"),
                        textInput("h_Dif2", "Precisión a Utilizar (h)")),
                    actionButton("DiferFinCentr2_Solver", "Calcular Derivada"),
                    textOutput("DiferFinCentr2_Out")),
            
            # Diferenciación - Diferencias Progresivas
            tabItem("DiferFinProg",
                    h1("Diferencia Finita Progresiva"),
                    h3("Función que calcula la derivada de una función 'f' en el punto 'x0' utilizando una diferencia finita progresiva."),
                    box(textInput("ecuacion_Dif3", "Ecuación a Evaluar"),
                        textInput("x0_Dif3", "Punto para el que se Calculará la Derivada Numérica (x0)"),
                        textInput("h_Dif3", "Precisión a Utilizar (h)")),
                    actionButton("DiferFinProg_Solver", "Calcular Derivada"),
                    textOutput("DiferFinProg_Out")),
            
            # ====================
            # DESCENSO DE GRADIENTE
            # ====================
            
            # Resolución de problema cuadrático
            tabItem("QP",
                    h1("Problema Cuadrático (QP)"),
                    h3("Minimización del problema cuadrático (QP) utilizando descenso de gradiente"),
                    box(
                        
                        # Matriz Q
                        div(style="width: 300px; display: inline-block;",
                            div(style="display: inline-block;vertical-align:center; font-size: 2rem; padding: 0 25px 0 25px;", p("Q = ")),
                            
                            # Fila 1
                            div(style="display: inline-block;vertical-align:top; width: 50px;", textInput("Q11", NULL)),
                            div(style="display: inline-block;vertical-align:top; width: 50px;", textInput("Q12", NULL)),
                            div(style="display: inline-block;vertical-align:top; width: 50px; margin: 0 20px 0 0;", textInput("Q13", NULL)),
                            # Fila 2
                            div(style="display: inline-block; width: 50px; padding: 0 0; margin: 0 0 0 80px;", textInput("Q21", NULL)),
                            div(style="display: inline-block; width: 50px; margin-bottom: 0;", textInput("Q22", NULL)),
                            div(style="display: inline-block; width: 50px; padding: 0 0; margin: 0 20px 0 0;", textInput("Q23", NULL)),
                            # Fila 3
                            div(style="display: inline-block; width: 50px; margin: 0 0 0 80px;", textInput("Q31", NULL)),
                            div(style="display: inline-block; width: 50px;", textInput("Q32", NULL)),
                            div(style="display: inline-block; width: 50px; margin: 0 20px 0 0;;", textInput("Q33", NULL)),
                        ),
                        
                        # Vector C
                        div(style="width: 220px; display: inline-block;",
                            div(style="display: inline-block;vertical-align:center; font-size: 2rem; padding: 0 25px 0 25px;", p("c = ")),
                            div(style="display: inline-block; width: 50px; padding: 0 0; margin: 0 20px 0 5px;", textInput("c1", NULL)),
                            div(style="display: inline-block; width: 50px; padding: 0 0; margin: 0 20px 0 80px;", textInput("c2", NULL)),
                            div(style="display: inline-block; width: 50px; padding: 0 0; margin: 0 20px 0 80px;", textInput("c3", NULL)),
                        ),
                        
                        # Vector x0
                        div(style="width: 220px; display: inline-block;",
                            div(style="display: inline-block;vertical-align:center; font-size: 2rem; padding: 0 25px 0 25px;", p("x0 = ")),
                            div(style="display: inline-block; width: 50px; padding: 0 0; margin: 0 20px 0 -5px;", textInput("x01", NULL)),
                            div(style="display: inline-block; width: 50px; padding: 0 0; margin: 0 20px 0 80px;", textInput("x02", NULL)),
                            div(style="display: inline-block; width: 50px; padding: 0 0; margin: 0 20px 0 80px;", textInput("x03", NULL)),
                        ),
                        
                        hr(),
                        
                        textInput("eps_QP", "Precisión (epsilon)"),
                        textInput("N_QP", "Número de Iteraciones Máx (N)"),
                        selectInput("tipoSZ_QP", "Tipo de Step Size", c("Exacto" = "Exacto", "Constante" = "Constante", "Variable" = "Variable")),
                        textInput("valSZ_QP", "Valor de Step Size (Solo para Tipo 'Constante')")
                        
                        ),
                        
                    actionButton("QP_Solver", "Minimizar"),
                    tableOutput("QP_Out")),
            
            
            # Resolución de función de rosenbrock
            tabItem("Rosenbrock",
                    h1("Función de Rosenbrock"),
                    h3("Minimización de la función de Rosenbrock, una función caracterizada por contar con una curva patológica"),
                    box(
                        # Vector x0
                        div(style="width: 220px; display: inline-block;",
                            div(style="display: inline-block;vertical-align:center; font-size: 2rem; padding: 0 25px 0 25px;", p("x0 = ")),
                            div(style="display: inline-block; width: 50px; padding: 0 0; margin: 0 20px 0 -5px;", textInput("x01_Rosen", NULL)),
                            div(style="display: inline-block; width: 50px; padding: 0 0; margin: 0 20px 0 80px;", textInput("x02_Rosen", NULL)),
                        ),
                        
                        hr(),
                        
                        textInput("eps_Rosen", "Precisión (epsilon)"),
                        textInput("N_Rosen", "Número de Iteraciones Máx (N)"),
                        textInput("valSZ_Rosen", "Valor de Step Size (alpha)")
                        
                    ),
                    
                    actionButton("Rosen_Solver", "Minimizar"),
                    tableOutput("Rosen_Out")),
            
            # ====================
            # VARIANTES GRADIENT DESCENT
            # ====================
            
            # Solución cerrada
            tabItem("SC",
                    h1("Solución Cerrada"),
                    h3("Solución verdadera o cerrada al problema de regresión lineal con la forma Ax = b"),
                    box(textInput("SC_n", "Número de Muestras"),
                        textInput("SC_d", "Número de Dimensiones")),
                    actionButton("SCSolver", "Resolver"),
                    textOutput("salidaSC")),
            
            # Batch gradient descent
            tabItem("BGD",
                    h1("Batch Gradient Descent (BGD)"),
                    h3("Utilización de BGD para resolver una regresión lineal con la forma Ax = b"),
                    box(textInput("BGD_sz", "Step Size o Learning Rate"),
                        textInput("BGD_eps", "Precisión (epsilon)"),
                        textInput("BGD_maxIter", "Numero Máximo de Iteraciones (k_max)")),
                    actionButton("BGDSolver", "Resolver"),
                    tableOutput("salidaBGD")),
            
            # Stochastic Gradient Descent
            tabItem("SGD",
                    h1("Stochastic Gradient Descent (SGD)"),
                    h3("Utilización de SGD para resolver una regresión lineal con la forma Ax = b"),
                    box(textInput("SGD_sz", "Step Size o Learning Rate"),
                        textInput("SGD_eps", "Precisión (epsilon)"),
                        textInput("SGD_maxIter", "Numero Máximo de Iteraciones (k_max)")),
                    actionButton("SGDSolver", "Resolver"),
                    tableOutput("salidaSGD")),
            
            # Mini-batch Gradient Descent
            tabItem("MBGD",
                    h1("Mini-batch Gradient Descent (MBGD)"),
                    h3("Utilización de MBGD para resolver una regresión lineal con la forma Ax = b"),
                    box(textInput("MBGD_sz", "Step Size o Learning Rate"),
                        textInput("MBGD_eps", "Precisión (epsilon)"),
                        textInput("MBGD_maxIter", "Numero Máximo de Iteraciones (k_max)"),
                        textInput("MBGD_batchSize", "Batch Size")),
                    actionButton("MBGDSolver", "Resolver"),
                    tableOutput("salidaMBGD")),
            
            # ====================
            # BACKTRACKING LINE SEARCH
            # ====================
            
            # Gradient Descent
            tabItem("GDBLS",
                    h1("Gradient Descent con Backtracking Line Search (BLS)"),
                    h3("Utilización de descenso de gradiente, en conjunto con 'backtracking line search', para encontrar los mínimos de la función de Rosenbrock."),
                    box(
                        # Vector x0
                        div(style="width: 220px; display: inline-block;",
                            div(style="display: inline-block;vertical-align:center; font-size: 2rem; padding: 0 25px 0 25px;", p("x0 = ")),
                            div(style="display: inline-block; width: 50px; padding: 0 0; margin: 0 20px 0 -5px;", textInput("x01_GDBLS", NULL)),
                            div(style="display: inline-block; width: 50px; padding: 0 0; margin: 0 20px 0 80px;", textInput("x02_GDBLS", NULL)),
                        ),
                        
                        hr(),
                        
                        textInput("GDBLS_eps", "Precisión (epsilon)"),
                        textInput("GDBLS_maxIter", "Número de Iteraciones Máximas")
                        ),
                    actionButton("GDBLSSolver", "Resolver"),
                    tableOutput("salidaGDBLS")),
            
            # Newton
            tabItem("NBLS",
                    h1("Método de Newton con Backtracking Line Search (BLS)"),
                    h3("Utilización del método de newton, en conjunto con 'backtracking line search', para encontrar los mínimos de la función de Rosenbrock."),
                    box(
                        # Vector x0
                        div(style="width: 220px; display: inline-block;",
                            div(style="display: inline-block;vertical-align:center; font-size: 2rem; padding: 0 25px 0 25px;", p("x0 = ")),
                            div(style="display: inline-block; width: 50px; padding: 0 0; margin: 0 20px 0 -5px;", textInput("x01_NBLS", NULL)),
                            div(style="display: inline-block; width: 50px; padding: 0 0; margin: 0 20px 0 80px;", textInput("x02_NBLS", NULL)),
                        ),
                        
                        hr(),
                        
                        textInput("NBLS_eps", "Precisión (epsilon)"),
                        textInput("NBLS_maxIter", "Número de Iteraciones Máximas"),
                        textInput("NBLS_stepSize", "Step Size Override (Introducir un valor si no se desea utilizar BLS)")
                        ),
                    actionButton("NBLSSolver", "Resolver"),
                    tableOutput("salidaNBLS"))
            
        )
    )
)
