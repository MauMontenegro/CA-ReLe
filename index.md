<!DOCTYPE html>
<html>
  <head>
    <meta charset="UTF-8">
    <!-- MathJax -->
    <script type="text/javascript"
      src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
    </script>
  </head>

  <body>
 # CARL: Forest Fire with DQN

[ToC]

Este documento es un registro de avances y modificaciones respecto al proyecto CARL, el cuál consiste en utilizar técnicas de Reinforcement Learning (DQN, PPO, A2C) en la optimización de problemas relacionados con Automatas Celulares (CA). Actualmente se estudian dos ambientes principales: BullDozer y Helicoptero.

## Ambiente

Los ambientes consisten principalmente en una Grid de tamaño variable cuyas celdas tienen ciertos estados y las transiciones de estado están definidas por la dinámica del sistema. Existe un agente externo que puede modificar el estado de las celdas para obtener un comportamiento deseable en la evolución del ambiente y optimizar una función de costo arbitraria. El ambiente que se utiliza se encuentra en el siguiente repositorio: https://github.com/elbecerrasoto/gym-cellular-automata


<img src="https://i.imgur.com/QzV4Jxs.png" alt="drawing" width="50%"/>

<img src="https://i.imgur.com/gZnLBQw.png" alt="drawing" width="60%"/>

La Grid es cuadrada y los experimentos se llevan a cabo con tres tipos de celdas: celdas con fuego (fire cells), celdas de árbol (tree cells) y celdas quemadas (burned cells).

Los agentes tienen ciertas acciones restringidas a su tipo y la posición en la que se encuentren, cada acción gasta unidades de tiempo. El ambiente tiene untiempo de actualización, cuando el tiempo se agota, se realizan los cálculos de transición de celdas en toda la Grid.

- El helicóptero sólo puede moverse y apagar celdas de fuego.
- El Bulldozer puede moverse y cortar árboles.

El Autómata Celular trata de emular el esparcimiento de un incendio en un área boscosa, y necesitamos crear estrategias para nuestro agente que eviten el mayor número de árboles quemados. Dichas estrategias o políticas serán generadas por métodos de Reinforcement Learning como DQN, PPO y A2C.



## Deep Q-Networks (DQN)

https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

Gracias a que la información de nuestro ambiente y agente está codificada en una grid cuyas celdas y posición influyen en los features principales del problema, podemos hacer uso de DQN utilizando redes convolucionales para capturar la información espacial del autómata celular.

Consideramos tareas en las que nuestro agente actúa sobre nuestro ambiente $E$ en cada paso de tiempo y se genera una secuencia de acciones-observaciones-recompensas. Nuestro agente sólo observa el estado general de nuestro ambiente (estado de todas las celdas de la grid y posición del agente), a dicho estado lo llamaremos $x_t \in R^d$. Al realizar una acción sobre $E$ recibimos una recompensa $r_t$. Esta recompensa depende de la secuencia de acciones y puede verse reflejada hasta mucho tiempo en el futuro.

A diferencia de la configuración que se usa normalmente al resolver juegos de Atari, en esta propuesta conocemos la informaciòn interna del sistema como para considerar esta tarea MDP, pero para fines prácticos y aprovechar los beneficios de la generalización sólo brindaremos el estado de la grid y del agente (dejando de lado el tiempo de actualización y la velocidad del viento) convirtiendo a ésta tarea en un POMPD. Otra diferencia es que en experimentos iniciales no utilizamos un stack de transiciones como en el artículo original.




### Double DQN (DDQN)


### Prioritized Experience Replay


### Dueling Q-Networks

  </body>
</html>

