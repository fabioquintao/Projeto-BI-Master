# Projeto-BI-Master

# Reinforcement Learning Aplicado a Trading de Bitcoin

#### Aluno: [Fábio Quintão]([https://github.com/fabioquintao)
#### Orientadora: Evelyn Batista
---

Trabalho apresentado ao curso [BI MASTER](https://ica.puc-rio.ai/bi-master) como pré-requisito para conclusão de curso e obtenção de crédito na disciplina "Projetos de Sistemas Inteligentes de Apoio à Decisão".

- [Link para o código](https://github.com/fabioquintao/Projeto-BI-Master/blob/main/RL.ipynb). 

## Resumo

Este projeto apresenta uma aplicação do Aprendizado por Reforço (Reinforcement Learning) aplicado a trading de criptomoeda. O Aprendizado por Reforço é uma área de estudo no campo de Machine Learning, onde um agente aprende a tomar decisões otimizadas através de interações com um ambiente. Neste paradigma, o agente executa ações e recebe recompensas ou penalidades com base nos resultados dessas ações. O objetivo do agente é maximizar a soma das recompensas ao longo do tempo.
Este projeto foca na aplicação de RL, especificamente utilizando o algoritmo Proximal Policy Optimization (PPO), para desenvolver estratégias de trading. O PPO, implementado através da biblioteca Stable Baselines, foi escolhido por sua eficácia em ambientes de alta incerteza e complexidade, como é o caso dos mercados financeiros.

###  Proximal Policy Optimization (PPO)

O termo "Proximal" refere-se à abordagem do algoritmo em manter as novas políticas de decisão próximas às políticas antigas durante o processo de aprendizado. Isso ajuda a evitar mudanças drásticas que podem ser prejudiciais e garante um aprendizado mais estável. O PPO opera com base em políticas de decisão, que são mapeamentos de estados percebidos do ambiente para ações a serem tomadas. Estas políticas são frequentemente estocásticas, o que significa que, para um dado estado, o algoritmo produz uma distribuição de probabilidade sobre as ações possíveis, em vez de uma única ação determinística. 

No trading de criptomoedas, por exemplo, o ambiente de mercado é altamente incerto e volátil. O PPO, ao lidar com este ambiente, aprende uma política que não apenas escolhe a melhor ação com base no estado atual do mercado, mas também considera a incerteza inerente. Isso significa que, mesmo em condições de mercado semelhantes, as ações tomadas pelo agente podem variar, refletindo a natureza estocástica da política aprendida. Em ambientes de trading, onde as condições podem mudar rapidamente, a capacidade de variar ações aumenta a adaptação e potencialmente melhora o desempenho.

Exploração e Explotação: Em Aprendizado por Reforço, um aspecto crucial é equilibrar a exploração (experimentar novas ações para descobrir suas recompensas) e a explotação (usar o conhecimento adquirido para tomar as melhores ações). A abordagem estocástica do PPO ajuda a manter este equilíbrio, permitindo que o modelo explore diferentes ações de forma probabilística.

Tecnicamente, o PPO realiza atualizações de política de uma maneira que equilibra eficientemente a exploração e a explotação, evitando grandes desvios que podem resultar em desempenho instável. Isso é alcançado através de várias características-chave:

1. **Atualização Limitada de Política**: O PPO limita a extensão das atualizações de política usando uma função objetivo clipada. A razão de probabilidade das políticas, `r_t(θ)`, é clipada dentro de um intervalo `[1 - ε, 1 + ε]`, onde `ε` é um hiperparâmetro. Isso assegura que as atualizações da política não se desviem drasticamente da política anterior, promovendo um aprendizado estável.

2. **Estimação de Vantagem Generalizada (GAE)**: O PPO utiliza GAE para calcular a vantagem de uma ação, que ajuda a balancear a variância e o viés nas estimativas de recompensa, onde `δ_t` é o erro de diferença temporal (TD) e `γ` e `λ` são hiperparâmetros.

3. **Função de Valor**: Paralelamente à política, o PPO treina uma função de valor usando erro médio quadrátco para estimar os retornos futuros.

### As principais etapas do projeto são:

*Preparação dos Dados*: Foram utilizados dados históricos do Bitcoin, disponíveis através do Yahoo Finance, abrangendo um período de 84 meses. Estes dados são usados para criar um ambiente de treinamento realista para o agente de RL. Os dados históricos incluem preço de abertura, preço de fechamento, alta e baixa do dia, volume de negociações, entre outros.
   
*Estratégia de Trading Personalizada*: Foi definida uma estratégia de trading chamada "Momentum and Volatility", que inclui uma série de indicadores técnicos. Estes indicadores são usados para analisar o mercado e tomar decisões informadas de trading.

***Indicadores Técnicos Utilizados***:

*Médias Móveis Simples (SMA) de 50 e 200 dias*: Estas médias são usadas para identificar tendências de longo e curto prazo no mercado. Uma SMA de 50 dias ajuda a entender o momentum de curto prazo, enquanto a SMA de 200 dias é frequentemente usada para identificar a tendência de longo prazo.

*Bandas de Bollinger (BBANDS) com um período de 20*: Este indicador é usado para medir a volatilidade do mercado e identificar overbought (sobrecomprado) ou oversold (sobrevendido) condições.

*Índice de Força Relativa (RSI)*: Um indicador de momentum que mede a velocidade e a mudança dos movimentos de preço. O RSI é comumente usado para identificar condições de sobrecompra ou sobrevenda no mercado.

*Convergência e Divergência de Médias Móveis (MACD)*: Este indicador ajuda a identificar mudanças de tendência no mercado através da comparação entre duas médias móveis de diferentes períodos.

*Média Móvel Simples de Volume (Volume SMA) de 20 dias*: Fornece insights sobre o volume de negociação, que é um importante indicador da força de uma tendência.
A combinação destes indicadores técnicos permite ao modelo aprender a identificar padrões e a tomar decisões de trading mais informadas e baseadas em dados.

*Configuração do Ambiente de Trading*: Foi criado um ambiente simulado baseado na biblioteca gym, que reflete o mercado de trading de criptomoedas. Este ambiente permite ao agente aprender e desenvolver estratégias de maneira controlada e iterativa. O ambiente proporciona uma plataforma para o agente experimentar diferentes estratégias, aprender com as interações e ajustar suas ações com base nos resultados obtidos.
   
O benchmark escolhido para este projeto é a estratégia de Buy-and-Hold. Esta estratégia consiste basicamente em comprar ativos e mantê-los por um longo período, independentemente das flutuações do mercado. O objetivo principal é permitir que o agente de RL desenvolva uma estratégia que não apenas aprenda a navegar pela volatilidade do mercado de criptomoedas, mas que também seja capaz de superar o retorno do benchmark.
 
Treinamento e Avaliação do Agente: O agente é treinado com base nos dados históricos do Bitcoin e avaliado através de métricas de backtesting, com o objetivo de testar a eficácia e a validade da estratégia desenvolvida.Este treinamento envolve o ajuste iterativo das políticas de decisão do agente com o objetivo de maximizar as recompensas.
   
### Métricas de Backtesting: 

•	*Análise de Retorno Total*: Avalia o ganho ou perda total gerado pela estratégia ao longo do período de teste.

•	*Drawdown Máximo*: Mede a maior queda da estratégia, fornecendo uma indicação do risco de perdas significativas.

•	*Sharpe Ratio*: Compara o retorno ajustado ao risco da estratégia, oferecendo uma perspectiva sobre sua eficiência em termos de geração de retorno por unidade de risco.

•	Uso de Simulações em Dados de Validação: Para assegurar a robustez e a aplicabilidade da estratégia em diferentes cenários de mercado, o processo de backtesting é realizado em um conjunto de dados de validação. Este conjunto é separado dos dados utilizados durante o treinamento, permitindo uma avaliação imparcial da estratégia. As características deste processo incluem:

•	Realização de 1000 Simulações: Dada a natureza estocástica do algoritmo PPO, são realizadas 1000 simulações para capturar a variabilidade nos resultados. Cada simulação pode apresentar trajetórias de trading ligeiramente diferentes, mesmo em condições de mercado semelhantes.

•	Cálculo da Média das Métricas: A média das métricas de todas as simulações é calculada para avaliar a consistência geral da estratégia.

•	Avaliação da Generalização da Estratégia: Utilizar dados de validação ajuda a confirmar se a estratégia desenvolvida é generalizável e eficaz fora do conjunto de dados de treinamento. Isso é crucial para garantir que a estratégia não esteja superajustada (overfitting) aos dados de treinamento e possa se adaptar a novos dados e condições de mercado.

### Análise Comparativa e Visualização:

Foi realizada uma comparacão do desempenho da estratégia contra um benchmark de mercado.  Utilizando a função backtest_strategy_ensemble, a estratégia de trading é testada utilizando a média dos resultados de 1000 simulações. 

•	Cálculo dos Retornos Cumulativos: A função calcula os retornos cumulativos da estratégia e do benchmark.

•	Visualização Gráfica: As séries de retornos cumulativos tanto da estratégia quanto do benchmark são plotadas em um gráfico. Esta visualização fornece uma comparação entre performance da estratégia de trading e o benchmark ao longo do tempo.

•	Análise de Desempenho: Através da visualização, é possível analisar não apenas o retorno total, mas também a volatilidade e a estabilidade da estratégia em comparação com o benchmark. Por exemplo, uma estratégia que exibe menos volatilidade e menores drawdowns em relação ao benchmark pode ser considerada mais favorável, mesmo que o retorno total seja semelhante.

 A tabela abaixo mostra os resultados da estratágia vs o benchmark:

         
|                         |   Strategy |   Benchmark |
|:------------------------|-----------:|------------:|
| Total Return (%)        |   -11.894  |    -18.3247 |
| Max Drawdown (%)        |    80.5798 |    110.08   |
| Annualized Sharpe Ratio |    -0.1184 |     -0.1585 |


O gráfico abaixo mostra os retornos cumulativos da estratégia vs o benchmark:

  ![image](https://github.com/fabioquintao/Projeto-BI-Master/assets/76189229/5d7e808c-59db-4e45-aa12-fa2cf4c352ff)


Otimização dos Hiperparâmetros
Integração com Optuna: O código integra o modelo com o Optuna para realizar a otimização de hiperparâmetros. Optuna automatiza o processo de experimentar diferentes combinações de hiperparâmetros e identificar as que oferecem o melhor desempenho.
Hiperparâmetros Selecionados para Otimização:

•	*Taxa de Aprendizado (learning_rate)*: Determina o tamanho dos ajustes feitos aos pesos da rede neural durante o treinamento. 

•	*Fator de Desconto (gamma)*: Este parâmetro equilibra a importância das recompensas imediatas versus futuras. 

•	*GAE Lambda (gae_lambda)*: Usado no cálculo do Generalized Advantage Estimator, um método para reduzir a variância dos estimadores de vantagem, melhorando a estabilidade do treinamento.

Após concluir a otimização, o Optuna fornece a configuração de hiperparâmetros que resultou no melhor desempenho, orientando a escolha final dos parâmetros para o modelo.

|                         |   Strategy |   Benchmark |
|:------------------------|-----------:|------------:|
| Total Return (%)        |  -18.2189  |  -18.3247   |
| Max Drawdown (%)        |  110.079   |   110.08    |
| Annualized Sharpe Ratio |  -0.1575   |   -0.1585   |

### Os resultados obtidos após a otimização podem ser devido a fatores como:

1. **Equilíbrio entre Exploração e Explotação**: A otimização de hiperparâmetros pode ter levado a um equilíbrio diferente entre exploração  e explotação. Isso pode resultar em um modelo que é menos propenso a assumir riscos ou a explorar novas estratégias que poderiam ter gerado retornos mais altos.

2. **Sobreajuste aos Dados de Treinamento (Overfitting)**: Uma possibilidade é que o modelo otimizado tenha se ajustado demais aos dados de treinamento. Isso significa que ele pode ter se tornado muito bom em prever as condições específicas dos dados de treinamento, mas menos eficaz em generalizar para novos dados ou condições de mercado.

3. **Complexidade do Modelo vs. Ruído do Mercado**: Em mercados financeiros, especialmente em criptomoedas, há um alto grau de ruído e imprevisibilidade. Um modelo mais complexo ou "ajustado" nem sempre é sinônimo de melhor desempenho, especialmente se o mercado em si é altamente volátil e impulsionado por fatores externos que não podem ser capturados por indicadores técnicos.

4. **Natureza Estocástica do Algoritmo PPO**: O algoritmo PPO é estocástico por natureza, o que significa que pode haver uma variação inerente nos resultados de cada simulação. A otimização pode ter levado a um conjunto de hiperparâmetros que, embora teoricamente ideal, na prática não produz consistentemente melhores resultados devido à variabilidade do mercado.

5. **Limitações dos Indicadores Técnicos**: Os indicadores técnicos usados podem não capturar completamente a complexidade e os fatores subjacentes que afetam os preços das criptomoedas. Assim, a estratégia baseada nesses indicadores, mesmo otimizada, pode ter limitações inerentes.

