library(shiny)
library(shinydashboard)

# Define UI for application that draws a histogram
dashboardPage(
    dashboardHeader(title = "Algoritmos en DS"),
    dashboardSidebar(
        sidebarMenu(
            menuItem("Ceros", tabName = "Ceros",
                     menuSubItem("Bisección", tabName = "Biseccion"),
                     menuSubItem("Newton-Raphson", tabName = "Newton")),
            menuItem("Derivación", tabName = "Derivacion",
                     menuSubItem("Diferencia Finita Centrada 1", tabName = "DiferFinCentr1"),
                     menuSubItem("Diferencia Finita Centrada 2", tabName = "DiferFinCentr2"),
                     menuSubItem("Diferencia Finita Progresiva", tabName = "DiferFinProg"))
        )
    ),
    
    dashboardBody(
        tabItems(
            tabItem("Newton",
                    h1("Método de Newton-Raphson"),
                    h3("Rutina para obtener los puntos estacionarios de una ecuación con la forma F(x) = 0 a través del método de Newton-Raphson."),
                    box(textInput("ecuacionNew", "Ecuación a Evaluar"),
                        textInput("x0New", "Solucion Inicial de Ecuación (X0)"),
                        textInput("maxIterNew", "Numero Máximo de Iteraciones (k_max)"),
                        textInput("epsilonNew", "Precision a Alcanzar (epsilon)")),
                    actionButton("nwtSolver", "Resolver"),
                    tableOutput("salidaNewton")),
            
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
            
            tabItem("DiferFinCentr1",
                    h1("Diferencia Finita Centrada 1"),
                    h3("Función que calcula la derivada de una función 'f' en el punto 'x0' utilizando dos miembros de la aproximación de Taylor."),
                    box(textInput("ecuacion_Dif1", "Ecuación a Evaluar"),
                        textInput("x0_Dif1", "Punto para el que se Calculará la Derivada Numérica (x0)"),
                        textInput("h_Dif1", "Precisión a Utilizar (h)")),
                    actionButton("DiferFinCentr1_Solver", "Calcular Derivada"),
                    textOutput("DiferFinCentr1_Out")),
            
            tabItem("DiferFinCentr2",
                    h1("Diferencia Finita Centrada 2"),
                    h3("Función que calcula la derivada de una función 'f' en el punto 'x0' utilizando cuatro miembros de la aproximación de Taylor."),
                    box(textInput("ecuacion_Dif2", "Ecuación a Evaluar"),
                        textInput("x0_Dif2", "Punto para el que se Calculará la Derivada Numérica (x0)"),
                        textInput("h_Dif2", "Precisión a Utilizar (h)")),
                    actionButton("DiferFinCentr2_Solver", "Calcular Derivada"),
                    textOutput("DiferFinCentr2_Out")),
            
            tabItem("DiferFinProg",
                    h1("Diferencia Finita Progresiva"),
                    h3("Función que calcula la derivada de una función 'f' en el punto 'x0' utilizando una diferencia finita progresiva."),
                    box(textInput("ecuacion_Dif3", "Ecuación a Evaluar"),
                        textInput("x0_Dif3", "Punto para el que se Calculará la Derivada Numérica (x0)"),
                        textInput("h_Dif3", "Precisión a Utilizar (h)")),
                    actionButton("DiferFinProg_Solver", "Calcular Derivada"),
                    textOutput("DiferFinProg_Out"))
        )
    )
)
