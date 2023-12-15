# Projeto-BI-Master

# Reinforcement Learning Aplicado a Trading de Bitcoin

#### Aluno: [Fábio Quintão]([https://github.com/fabioquintao)
#### Orientadora: Evelyn Batista
---

Trabalho apresentado ao curso [BI MASTER](https://ica.puc-rio.ai/bi-master) como pré-requisito para conclusão de curso e obtenção de crédito na disciplina "Projetos de Sistemas Inteligentes de Apoio à Decisão".

- [Link para o código](https://github.com/fabioquintao/Projeto-BI-Master/blob/main/RL_for%20_trading.ipynb)

## RESUMO

Este projeto apresenta uma aplicação do Aprendizado por Reforço (Reinforcement Learning) para trading de criptomoeda. O Aprendizado por Reforço é uma área de estudo no campo de Machine Learning, onde um agente aprende a tomar decisões otimizadas através de interações com um ambiente. Neste paradigma, o agente executa ações e recebe recompensas ou penalidades com base nos resultados dessas ações. O objetivo do agente é maximizar a soma das recompensas ao longo do tempo.
Este projeto foca na aplicação de RL, especificamente utilizando o algoritmo Proximal Policy Optimization (PPO), para desenvolver estratégias de trading. O PPO, implementado através da biblioteca Stable Baselines, foi escolhido por sua eficácia em ambientes de alta incerteza e complexidade, como é o caso dos mercados financeiros.


###  PROXIMAL POLICY OPTIMIZATION (PPO)


O termo "Proximal" refere-se à abordagem do algoritmo em manter as novas políticas de decisão próximas às políticas antigas durante o processo de aprendizado. O PPO opera com base em políticas de decisão, que são mapeamentos de estados percebidos do ambiente para ações a serem tomadas. Estas políticas são frequentemente estocásticas, o que significa que, para um dado estado, o algoritmo produz uma distribuição de probabilidade sobre as ações possíveis, em vez de uma única ação determinística. 

No trading de criptomoedas, por exemplo, o ambiente de mercado é altamente incerto e volátil. O PPO, ao lidar com este ambiente, aprende uma política que não apenas escolhe a melhor ação com base no estado atual do mercado, mas também considera a incerteza inerente. Isso significa que, mesmo em condições de mercado semelhantes, as ações tomadas pelo agente podem variar, refletindo a natureza estocástica da política aprendida. Em ambientes de trading, onde as condições podem mudar rapidamente, a capacidade de variar ações aumenta a adaptação e potencialmente melhora o desempenho.

**Exploração e Explotação**: Em Aprendizado por Reforço, um aspecto crucial é equilibrar a exploração (experimentar novas ações para descobrir suas recompensas) e a explotação (usar o conhecimento adquirido para tomar as melhores ações). A abordagem estocástica do PPO ajuda a manter este equilíbrio, permitindo que o modelo explore diferentes ações de forma probabilística.


### COMPONENTES PRINCIPAIS DO PPO:


**Rede de Políticas (Policy Network)**: Esta rede é responsável por tomar decisões. Ela mapeia estados do ambiente para ações, determinando como o agente deve se comportar em determinadas situações. Nesse modelo foi utilzada a "MLP Policy", ou Política de Perceptron Multi-Camadas, que se caracteriza por uma rede de múltiplas camadas de neurônios, cada camada conectada à seguinte, permitindo aprender representações complexas dos dados de entrada.

**Rede de Valor (Value Network)**: Esta rede avalia a qualidade de um estado (ou ação) ao prever a quantidade de recompensa que o agente pode esperar receber a partir daquele estado. Paralelamente à política, o algoritmo treina uma função de valor usando erro médio quadrátco para estimar os retornos futuros.

**Aprendizado Iterativo**: O PPO funciona através de iterações. Em cada iteração, o agente interage com o ambiente, coletando dados sobre suas experiências.

**Atualização de Política**: Com base nos dados coletados, o PPO ajusta os parâmetros da rede de políticas. O objetivo é melhorar a política de ação do agente de forma a maximizar a recompensa esperada. 

**Restrição de Atualização de Política**: O PPO inclui um mecanismo que limita o tamanho das atualizações de política, o que evita mudanças drásticas e potencialmente prejudiciais. Isso é feito calculando a razão entre a nova política e a política antiga e restringindo-a dentro de um intervalo determinado. Tecnicamente, o algoritmo limita a extensão dessas atualizações usando uma função objetivo clipada. A razão de probabilidade das políticas `r_t(θ)` é clipada dentro de um intervalo `[1 - ε, 1 + ε]` onde ε é um hiperparâmetro. Isso assegura que as atualizações da política não se desviem drasticamente da política anterior, promovendo um aprendizado estável. `r_t(θ)` é a razão entre a probabilidade de uma ação sob a política atual e a probabilidade da mesma ação sob a política antiga. Se `r_t(θ)` estiver fora do intervalo `[1 - ε, 1 + ε]` onde ε é um hiperparâmetro pequeno (como 0.1 ou 0.2), a função de perda é clipada para reduzir o incentivo para mudar a política nessa direção.

**Estimação de Vantagem Generalizada (GAE)**: O PPO utiliza GAE para calcular a vantagem de uma ação, que ajuda a balancear a variância e o viés nas estimativas de recompensa, onde `δ` é o erro de diferença temporal (TD) e `λ` é um hiperparâmetro que equilibra a variância e o viés nas estimativas de vantagem. Um valor alto de `λ` pode aumentar a variância mas reduzir o viés, e vice-versa.


**As principais etapas do projeto são**:


### PREPARAÇÃO DOS DADOS:

Foram utilizados dados históricos do Bitcoin, disponíveis através do Yahoo Finance, abrangendo um período de 84 meses. Estes dados são usados para criar um ambiente de treinamento realista para o agente de RL. Os dados históricos incluem preço de abertura, preço de fechamento, alta e baixa do dia e volume de negociações.
   
**Estratégia de Trading Personalizada**: Foi definida uma estratégia de trading chamada "Momentum and Volatility", que inclui uma série de indicadores técnicos. Estes indicadores são usados para analisar o mercado e tomar decisões informadas de trading.


**Indicadores Técnicos Utilizados**:


- **Médias Móveis Simples (SMA) de 50 e 200 dias**: Estas médias são usadas para identificar tendências de longo e curto prazo no mercado. Uma SMA de 50 dias ajuda a entender o momentum de curto prazo, enquanto a SMA de 200 dias é frequentemente usada para identificar a tendência de longo prazo.

- **Bandas de Bollinger (BBANDS) com um período de 20**: Este indicador é usado para medir a volatilidade do mercado e identificar overbought (sobrecomprado) ou oversold (sobrevendido) condições.

- **Índice de Força Relativa (RSI)**: Um indicador de momentum que mede a velocidade e a mudança dos movimentos de preço. O RSI é comumente usado para identificar condições de sobrecompra ou sobrevenda no mercado.

- **Convergência e Divergência de Médias Móveis (MACD)**: Este indicador ajuda a identificar mudanças de tendência no mercado através da comparação entre duas médias móveis de diferentes períodos.

- **Média Móvel Simples de Volume (Volume SMA) de 20 dias**: Fornece insights sobre o volume de negociação, que é um importante indicador da força de uma tendência.

A combinação destes indicadores técnicos permite ao modelo aprender a identificar padrões e a tomar decisões de trading mais informadas e baseadas em dados.

**Configuração do Ambiente de Trading**: Foi criado um ambiente simulado baseado na biblioteca Gym, que reflete o mercado de trading de criptomoedas. Este ambiente permite ao agente aprender e desenvolver estratégias de maneira controlada e iterativa. O ambiente proporciona uma plataforma para o agente experimentar diferentes estratégias, aprender com as interações e ajustar suas ações com base nos resultados obtidos.
   
O benchmark escolhido para este projeto é a estratégia de Buy-and-Hold. Esta estratégia consiste basicamente em comprar ativos e mantê-los por um longo período, independentemente das flutuações do mercado. *O objetivo principal é permitir que o agente de RL desenvolva uma estratégia que não apenas aprenda a navegar pela volatilidade do mercado de criptomoedas, mas que também seja capaz de superar o retorno do benchmark*.

 
### TREINAMENTO E AVALIAÇÃO DO AGENTE:


O agente é treinado com base nos dados históricos do Bitcoin e avaliado através de métricas de backtesting, com o objetivo de testar a eficácia e a validade da estratégia desenvolvida. Este treinamento envolve o ajuste iterativo das políticas de decisão do agente com o objetivo de maximizar as recompensas.
   
**Métricas de Backtesting**: 

- **Retorno Total**: Avalia o ganho ou perda total gerado pela estratégia ao longo do período de teste.

- **Drawdown Máximo**: Mede a a maior queda percentual entre um pico e um vale subsequente no período, fornecendo uma indicação do risco de perdas significativas.

- **Sharpe Ratio**: Compara o retorno ajustado ao risco da estratégia, oferecendo uma perspectiva sobre sua eficiência em termos de geração de retorno por unidade de risco.

**Uso de Dados de Validação**: Para assegurar a robustez e a aplicabilidade da estratégia em diferentes cenários de mercado, o processo de backtesting é realizado em um conjunto de dados de validação. Este conjunto é separado dos dados utilizados durante o treinamento, permitindo uma avaliação imparcial da estratégia. As características deste processo incluem:

- **Simulações**: Dada a natureza estocástica do algoritmo PPO, são realizadas 100 simulações para capturar a variabilidade nos resultados. Cada simulação pode apresentar trajetórias de trading ligeiramente diferentes, mesmo em condições de mercado semelhantes.

- **Cálculo da Média das Métricas**: A média das métricas de todas as simulações é calculada para avaliar a consistência geral da estratégia.

- **Avaliação da Generalização da Estratégia**: Utilizar dados de validação ajuda a confirmar se a estratégia desenvolvida é generalizável e eficaz fora do conjunto de dados de treinamento. Isso é crucial para garantir que a estratégia não esteja superajustada (overfitting) aos dados de treinamento e possa se adaptar a novos dados e condições de mercado.


### ANÁLISE DOS RESULTADOS:


Foi realizada uma comparacão do desempenho da estratégia contra o benchmark. Utilizando a função *backtest_strategy*, a estratégia de trading é testada utilizando a média dos resultados de 100 simulações. 

- **Cálculo dos Retornos Cumulativos**: A função calcula os retornos cumulativos da estratégia e do benchmark.

- **Visualização Gráfica**: As séries de retornos cumulativos são plotadas em um gráfico. Esta visualização fornece uma comparação de performance ao longo do tempo.

- **Análise de Desempenho**: Através da visualização, é possível analisar não apenas o retorno total, mas também a volatilidade e a estabilidade da estratégia em comparação com o benchmark. Por exemplo, uma estratégia que exibe menos volatilidade e menores drawdowns pode ser considerada mais favorável, mesmo que o retorno total seja semelhante.

 A tabela abaixo mostra os resultados:


         
|                         |   Strategy |   Benchmark |
|:------------------------|-----------:|------------:|
| Total Return (%)        |    25.1945 |    -21.6774 |
| Max Drawdown (%)        |   130.223  |    145.391  |
| Annualized Sharpe Ratio |     0.1576 |     -0.1119 |


O gráfico abaixo mostra os retornos cumulativos:


![image](https://github.com/fabioquintao/Projeto-BI-Master/assets/76189229/5b3970e7-8d4c-4547-929a-dc5eff5a6daf)



### OTIMIZAÇÃO DOS HIPERPARÂMETROS:


**Integração com Optuna**: O código integra o modelo com o Optuna para realizar a otimização. Optuna automatiza o processo de experimentar diferentes combinações e identificar as que oferecem o melhor desempenho. Foi definida uma função *(optimize_ppo)* que cria uma instância do modelo com um conjunto de hiperparâmetros e avaliava seu desempenho (através da função *evaluate_model*).

Hiperparâmetros Selecionados para Otimização:

- **Taxa de Aprendizado (learning_rate)**: Determina o tamanho dos ajustes feitos aos pesos da rede neural durante o treinamento. 

- **Fator de Desconto (gamma)**: Equilibra a importância das recompensas imediatas versus futuras. 

- **GAE Lambda (gae_lambda)**: Usado no cálculo do Generalized Advantage Estimator, método para equilibrar o viés e a variância dos estimadores de vantagem, melhorando a estabilidade do treinamento.
  
- **Entropy Coefficient (entropy_coefficient)**: Ajusta o equilíbrio entre exploração e explotação.
  
- **Clip Range (clip_range)**: Controla o limite das mudanças na política de decisão do modelo.
  
- **Value Function Coefficient (value_function_coefficient)**: Define a importância da função de valor na função de perda total.

**Processo de Otimização**: 

O Optuna realiza várias tentativas (trials), cada uma com um conjunto diferente de hiperparâmetros. Ele utiliza algoritmos avançados para escolher os hiperparâmetros em cada trial, como o Tree-structured Parzen Estimator (TPE). O TPE modela a relação entre hiperparâmetros e a pontuação da função objetivo. Ele usa essa modelagem para prever quais conjuntos podem resultar em melhor desempenho, focando as futuras trials nessas áreas do espaço de hiperparâmetros. Enquanto o Grid Search explora  de forma exaustiva e o Random Search faz isso de maneira aleatória, o Optuna adota uma abordagem mais inteligente e eficiente, aprendendo com os resultados das tentativas anteriores para direcionar a busca para as regiões mais promissoras, oferecendo um balanço mais eficaz.

- **Pruning (Poda)**:
O Optuna oferece uma característica chamada "pruning", que é uma forma de parar prematuramente uma trial que não parece promissora.
Durante uma trial, se certos critérios intermediários indicam que essa configuração de hiperparâmetros provavelmente não resultará em um bom desempenho, a trial é "podada" (ou seja, interrompida) para economizar recursos. 

- **Seleção dos Melhores Hiperparâmetros**: Após várias tentativas, o Optuna identifica os hiperparâmetros que maximizam a função objetivo, neste caso *o sharpe ratio da estratégia*.

- **Treinamento do Modelo Otimizado**: Após a conclusão do processo de otimização, foi usado o *study.best_params* para obter o conjunto que resultou no melhor desempenho. Com esse conjunto, uma nova instância do modelo foi criada e treinada no mesmo ambiente de aprendizado. Esta instância foi configurada especificamente com os valores otimizados:
  
  

|                            |   Value |
|:---------------------------|--------:|
| learning_rate              |  0.0324 |
| gamma                      |  0.9577 |
| gae_lambda                 |  0.8922 |
| entropy_coefficient        |  0.0039 |
| clip_range                 |  0.2618 |
| value_function_coefficient |  0.9287 |



Abaixo estão os resultados da estratégia otimizada:



|                         |   Strategy |   Benchmark |
|:------------------------|-----------:|------------:|
| Total Return (%)        |    56.8488 |    -21.6774 |
| Max Drawdown (%)        |   102.075  |    145.391  |
| Annualized Sharpe Ratio |     0.2891 |     -0.1119 |


![image](https://github.com/fabioquintao/Projeto-BI-Master/assets/76189229/8cc5756a-29b1-43df-8562-a27ef18e3b65)


### CONCLUSÃO:


O modelo apresentado demonstra um potencial para desenvolver estratégias de trading adaptativas. O agente apresenta uma capacidade de adaptação em posicões de compra e venda com base nas recompensas que recebe como resultado de suas ações. Esse ajuste dinâmico da posição permite que o agente aprenda com as experiências passadas e ajuste seu nível de exposição ao mercado de acordo com o desempenho atual. A utilização de dados de validação no backtest contribui para mitigar o overfitting, reduzindo a probabilidade de a estratégia ser excessivamente adaptada aos dados de treinamento e, assim, aumentando a confiabilidade para lidar com diferentes cenários de mercado.

Embora o ambiente de simulação seja valioso para o treinamento inicial, os mercados de criptomoedas são altamente voláteis, o que desafia a capacidade do modelo de se adaptar rapidamente a mudanças inesperadas nas condições de mercado. Além disso, a aplicação do modelo no mundo real enfrenta desafios adicionais, como custos de transação, latência e questões de execução de ordens. A implementação bem-sucedida de uma estratégia de posicionamento dinâmico requer uma abordagem cuidadosa para equilibrar a busca por oportunidades de lucro com a gestão adequada de riscos e custos, além da capacidade de responder de forma eficaz a todas essas variáveis complexas no mundo real do trading.



