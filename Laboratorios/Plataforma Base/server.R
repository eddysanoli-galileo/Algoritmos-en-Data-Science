
library(shiny)
library(reticulate)

# Se especifica la versión de python a usar
use_python("~/AppData/Local/Programs/Python/Python39")

# Se define el archivo del que vienen las funciones de Python
source_python("algoritmos.py")


shinyServer(function(input, output) {
    
    # ================================
    # EVALUACIÓN DE EVENTOS
    # ================================
    
    # ----------------
    # Cálculo de Ceros
    # ----------------
    
    # Método de Newton-Raphson
    newtonCalculate = eventReactive(input$nwtSolver, {
        
        # Se convierten todos los inputs en string a números
        in_EcuacionStr = input$ecuacionNew[1]
        in_X0 = as.numeric(input$x0New[1])
        in_MaxIter = as.numeric(input$maxIterNew[1])
        in_Epsilon = as.numeric(input$epsilonNew[1])
        
        # Se convierte la ecuación en string en una función lambda
        # "proc_Ecuacion" consiste de un objeto con dos elementos. El primero contiene la función lambda
        proc_Ecuacion = parseEquation(in_EcuacionStr)
        in_Ecuacion = proc_Ecuacion[[1]]
        
        # Se imprime la ecuación convertida a formato de Python
        print("Método de Newton-Raphson:")
        print(proc_Ecuacion[[2]])
        
        # Se ejecuta el método de Newton.
        # El output de la función consiste de la solución "xk" y la tabla con record de iteraciones
        solOutput = SolverNewton(in_Ecuacion, in_X0, in_MaxIter, in_Epsilon)
        
        # Se retorna la tabla con el record de iteraciones
        return(solOutput[[2]])
    })
    
    # Método de Bisección
    biseccionCalculate = eventReactive(input$bisSolver, {
        
        # Se convierten todos los inputs en string a números
        in_EcuacionStr = input$ecuacionBis[1]
        in_limInf = as.numeric(input$lim_infBis[1])
        in_limSup = as.numeric(input$lim_supBis[1])
        in_MaxIter = as.numeric(input$maxIterBis[1])
        in_Epsilon = as.numeric(input$epsilonBis[1])
        
        # Se convierte la ecuación en string en una función lambda
        # "proc_Ecuacion" consiste de un objeto con dos elementos. El primero contiene la función lambda
        proc_Ecuacion = parseEquation(in_EcuacionStr)
        in_Ecuacion = proc_Ecuacion[[1]]
        
        # Se imprime la ecuación convertida a formato de Python
        print("Método de Bisección:")
        print(proc_Ecuacion[[2]])
        
        # Se ejecuta el método de Newton.
        # El output de la función consiste de la solución "xk" y la tabla con record de iteraciones
        solOutput = SolverBiseccion(in_Ecuacion, in_limInf, in_limSup, in_MaxIter, in_Epsilon)
        
        # Se retorna la tabla con el record de iteraciones
        return(solOutput[[2]])
    })
    
    # ----------------
    # Diferenciación
    # ----------------
    
    # Diferencias Finitas Centradas 1
    diferFinit1Calculate = eventReactive(input$DiferFinCentr1_Solver, {
        
        # Se convierten todos los inputs en string a números (menos la ecuación)
        in_EcuacionStr = input$ecuacion_Dif1[1]
        in_x0 = as.numeric(input$x0_Dif1[1])
        in_h = as.numeric(input$h_Dif1[1])
        
        # Se convierte la ecuación en string en una función lambda
        # "proc_Ecuacion" consiste de un objeto con dos elementos. El primero contiene la función lambda
        proc_Ecuacion = parseEquation(in_EcuacionStr)
        in_Ecuacion = proc_Ecuacion[[1]]
        
        # Se imprime la ecuación convertida a formato de Python
        print("Diferencia Finita Centrada 1:")
        print(proc_Ecuacion[[2]])
        
        # Se calcula la derivada
        df = DiferFinitaCentrada1(in_Ecuacion, in_x0, in_h)
        
        # Se retorna la tabla de resultados
        return(df)
        
    })
    
    # Diferencias Finitas Centradas 2
    diferFinit2Calculate = eventReactive(input$DiferFinCentr2_Solver, {
        
        # Se convierten todos los inputs en string a números (menos la ecuación)
        in_EcuacionStr = input$ecuacion_Dif2[1]
        in_x0 = as.numeric(input$x0_Dif2[1])
        in_h = as.numeric(input$h_Dif2[1])
        
        # Se convierte la ecuación en string en una función lambda
        # "proc_Ecuacion" consiste de un objeto con dos elementos. El primero contiene la función lambda
        proc_Ecuacion = parseEquation(in_EcuacionStr)
        in_Ecuacion = proc_Ecuacion[[1]]
        
        # Se imprime la ecuación convertida a formato de Python
        print("Diferencia Finita Centrada 2:")
        print(proc_Ecuacion[[2]])
        
        # Se calcula la derivada
        df = DiferFinitaCentrada2(in_Ecuacion, in_x0, in_h)
        
        # Se retorna la tabla de resultados
        return(df)
        
    })
    
    # Diferencias Finitas Progresivas
    diferFinit3Calculate = eventReactive(input$DiferFinProg_Solver, {
        
        # Se convierten todos los inputs en string a números (menos la ecuación)
        in_EcuacionStr = input$ecuacion_Dif3[1]
        in_x0 = as.numeric(input$x0_Dif3[1])
        in_h = as.numeric(input$h_Dif3[1])
        
        # Se convierte la ecuación en string en una función lambda
        # "proc_Ecuacion" consiste de un objeto con dos elementos. El primero contiene la función lambda
        proc_Ecuacion = parseEquation(in_EcuacionStr)
        in_Ecuacion = proc_Ecuacion[[1]]
        
        # Se imprime la ecuación convertida a formato de Python
        print("Diferencia Finita Progresiva:")
        print(proc_Ecuacion[[2]])
        
        # Se calcula la derivada
        df = DiferFinitaProgresiva(in_Ecuacion, in_x0, in_h)
        
        # Se retorna la tabla de resultados
        return(df)
        
    })
    
    
    # ================================
    # RENDER DE SALIDAS A UI
    # ================================
    
    # Render Newton-Raphson
    output$salidaNewton = renderTable({
        newtonCalculate()
    }, digits = 8)
    
    # Render Bisección
    output$salidaBiseccion = renderTable({
        biseccionCalculate()
    }, digits = 8)
    
    # Render Diferencia Finita Centrada 1
    output$DiferFinCentr1_Out = renderText({
        diferFinit1Calculate()
    })
    
    # Render Diferencia Finita Centrada 2
    output$DiferFinCentr2_Out = renderText({
        diferFinit2Calculate()
    })
    
    # Render Diferencia Finita Progresiva
    output$DiferFinProg_Out = renderText({
        diferFinit3Calculate()
    })
    
    
})
