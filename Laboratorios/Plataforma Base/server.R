
library(shiny)
library(reticulate)

# Se especifica la versión de python a usar
use_python("~/AppData/Local/Programs/Python/Python39")

# Se define el archivo del que vienen las funciones de Python
source_python("algoritmos.py")


shinyServer(function(input, output, session) {
    
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
    
    # ----------------
    # Descenso Gradiente
    # ----------------
    
    # Descenso de Gradiente para QP
    QPCalculate = eventReactive(input$QP_Solver, {
        
        # CONSTRUCCIÓN DE MATRIZ Q
        
        # Se crea una matriz de 3x3 con unos
        Q = matrix(1, 3, 3)
        
        # Se susituyen los unos por los valores introducidos por el usuario
        Q[1,1] = as.numeric(input$Q11[1])
        Q[1,2] = as.numeric(input$Q12[1])
        Q[1,3] = as.numeric(input$Q13[1])
        Q[2,1] = as.numeric(input$Q21[1])
        Q[2,2] = as.numeric(input$Q22[1])
        Q[2,3] = as.numeric(input$Q23[1])
        Q[3,1] = as.numeric(input$Q31[1])
        Q[3,2] = as.numeric(input$Q32[1])
        Q[3,3] = as.numeric(input$Q33[1])
        
        # CONSTRUCCIÓN DE VECTOR C
        
        # Vector de 1x3 con unos
        c = matrix(1, 3, 1)
        
        # Se sustituyen los unos por los valores de usuario
        c[1,1] = as.numeric(input$c1[1])
        c[2,1] = as.numeric(input$c2[1])
        c[3,1] = as.numeric(input$c3[1])
        
        # CONSTRUCCIÓN DE VECTOR X0
        
        # Vector de 1x3 con unos
        x0 = matrix(1, 3, 1)
        
        # Se sustituyen los unos por los valores de usuario
        x0[1,1] = as.numeric(input$x01[1])
        x0[2,1] = as.numeric(input$x02[1])
        x0[3,1] = as.numeric(input$x03[1])
        
        # Se convierten en números los inputs restantes
        epsilon = as.numeric(input$eps_QP[1])
        
        N = as.numeric(input$N_QP[1])
        val_SZ = as.numeric(input$valSZ_QP[1])
        
        # El tipo de step-size se mantiene como string
        tipo_SZ = input$tipoSZ_QP[1]

        # Se calcula la derivada
        table = SolverQP(Q, c, x0, epsilon, N, tipo_SZ, val_SZ)
        
        # Se retorna la tabla de resultados
        return(table)
        
    })
    
    # Descenso de Gradiente para Función de Rosenbrock
    RosenCalculate = eventReactive(input$Rosen_Solver, {
        
        # CONSTRUCCIÓN DE VECTOR X0
        
        # Vector de 1x2 con unos
        x0 = matrix(1, 2, 1)
        
        # Se sustituyen los unos por los valores de usuario
        x0[1,1] = as.numeric(input$x01_Rosen[1])
        x0[2,1] = as.numeric(input$x02_Rosen[1])
        
        # Se convierten en números los inputs restantes
        epsilon = as.numeric(input$eps_Rosen[1])
        N = as.numeric(input$N_Rosen[1])
        val_SZ = as.numeric(input$valSZ_Rosen[1])
        
        # Se calcula la derivada
        table = SolverRosenbrock(x0, epsilon, N, val_SZ)
        
        # Se retorna la tabla de resultados
        return(table)
        
    })
    
    # ----------------
    # Variantes Descenso Gradiente
    # ----------------
    
    # Batch Gradient Descent
    SCCalculate = eventReactive(input$SCSolver, {
        
        # Se convierten en números los inputs restantes
        n = as.integer(input$SC_n[1])
        d = as.integer(input$SC_d[1])
        
        # Se calcula la solución cerrada
        x_star = solCerrada(n, d)
        
        # Se retorna el vector resultante
        return(x_star)
        
    })
    
    # Batch Gradient Descent
    BGDCalculate = eventReactive(input$BGDSolver, {
        
        # Se convierten en números los inputs restantes
        epsilon = as.numeric(input$BGD_eps[1])
        max_iter = as.numeric(input$BGD_maxIter[1])
        step_size = as.numeric(input$BGD_sz[1])
        
        # Se calcula el mínimo
        table = SolverLinReg("batch", epsilon, max_iter, step_size, 0)
        
        # Se retorna la tabla de resultados
        return(table)
        
    })
    
    # Stochastic Gradient Descent
    SGDCalculate = eventReactive(input$SGDSolver, {
        
        # Se convierten en números los inputs restantes
        epsilon = as.numeric(input$SGD_eps[1])
        max_iter = as.numeric(input$SGD_maxIter[1])
        step_size = as.numeric(input$SGD_sz[1])
        
        print("Stochastic Gradient Descent")
        
        # Se calcula el mínimo
        table = SolverLinReg("stochastic", epsilon, max_iter, step_size, 0)
        
        print("Done!")
        
        # Se retorna la tabla de resultados
        return(table)
        
    })
    
    # Mini-batch Gradient Descent
    MBGDCalculate = eventReactive(input$MBGDSolver, {
        
        # Se convierten en números los inputs restantes
        epsilon = as.numeric(input$MBGD_eps[1])
        max_iter = as.integer(input$MBGD_maxIter[1])
        step_size = as.numeric(input$MBGD_sz[1])
        batch_size = as.integer(input$MBGD_batchSize[1])
        
        print("Mini-batch Gradient Descent")
        
        # Se calcula el mínimo
        table = SolverLinReg("mini-batch", epsilon, max_iter, step_size, batch_size)
        
        print("Done!")
        
        # Se retorna la tabla de resultados
        return(table)
        
    })
    
    # ----------------
    # Backtracking Line Search
    # ----------------
    
    # Gradient Descent con Backtracking Line Search
    GDBLSCalculate = eventReactive(input$GDBLSSolver, {
        
        # CONSTRUCCIÓN DE VECTOR X0
        
        # Vector de 1x2 con unos
        x0 = matrix(1, 2, 1)
        
        # Se sustituyen los unos por los valores de usuario
        x0[1,1] = as.numeric(input$x01_GDBLS[1])
        x0[2,1] = as.numeric(input$x02_GDBLS[1])
        
        # Se convierten en números los inputs restantes
        epsilon = as.numeric(input$GDBLS_eps[1])
        max_iter = as.numeric(input$GDBLS_maxIter[1])
        
        # Se calcula el mínimo
        table = gradientDescentBLS(x0, epsilon, max_iter)
        
        # Se retorna la tabla de resultados
        return(table)
        
    })
    
    # Gradient Descent con Backtracking Line Search
    NBLSCalculate = eventReactive(input$NBLSSolver, {
        
        # CONSTRUCCIÓN DE VECTOR X0
        
        # Vector de 1x2 con unos
        x0 = matrix(1, 2, 1)
        
        # Se sustituyen los unos por los valores de usuario
        x0[1,1] = as.numeric(input$x01_NBLS[1])
        x0[2,1] = as.numeric(input$x02_NBLS[1])
        
        # Se convierten en números los inputs restantes
        epsilon = as.numeric(input$NBLS_eps[1])
        max_iter = as.numeric(input$NBLS_maxIter[1])
        alpha_override = as.numeric(input$NBLS_stepSize[1])
        
        # Si el valor de alpha es NA, se le asigna un valor de NULL o None
        # Si no se coloca esto, python toma el valor de alpha override como
        # NA y asigna un NaN a alpha.
        if (is.na(alpha_override) == TRUE){
            alpha_override = NULL 
        }
            
        print("Newton BLS")
        
        # Se calcula el mínimo
        table = NewtonBLS(x0, epsilon, max_iter, alpha_override)
        
        print("Done!")
        
        # Se retorna la tabla de resultados
        return(table)
        
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
    
    # Render Solución de QP
    output$QP_Out = renderTable({
        QPCalculate()
    })
    
    # Render Solución de Rosenbrock
    output$Rosen_Out = renderTable({
        RosenCalculate()
    })
    
    # Render Solución de Solución Exacta
    output$salidaSC = renderText({
        SCCalculate()
    })
    
    # Render Solución de Batch Gradient Descent
    output$salidaBGD = renderTable({
        BGDCalculate()
    })
    
    # Render Solución de Stochastic Gradient Descent
    output$salidaSGD = renderTable({
        SGDCalculate()
    })
    
    # Render Solución de Mini-batch Gradient Descent
    output$salidaMBGD = renderTable({
        MBGDCalculate()
    })
    
    # Render Solución de Gradient Descent con Backtracking Line Search
    output$salidaGDBLS = renderTable({
        GDBLSCalculate()
    })
    
    # Render Solución de Método de Newton con Backtracking Line Search
    output$salidaNBLS = renderTable({
        NBLSCalculate()
    })
    
    
})
