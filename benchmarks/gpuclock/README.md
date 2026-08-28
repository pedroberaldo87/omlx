# gpuclock — cronômetro de GPU por ciclo de decode (v8 F2.2)

O MLX não expõe tempo de GPU em Python (o PR #3978 foi fechado sem merge), e a
captura `.gputrace` é inviável com este checkpoint (inflou para 102 GB e foi
abortada). O único mecanismo por buffer que existe é `GPUStartTime`/`GPUEndTime`
do `MTLCommandBuffer`.

Este dylib troca (swizzle) o seletor que o MLX usa para criar buffers de comando
— `commandBufferWithUnretainedReferences` (`mlx/backend/metal/device.cpp:321`) —
e registra o par de tempos de cada buffer que completa. Assim dá para separar
**GPU trabalhando** de **GPU esperando o Python**, que é a pergunta que decide
onde mora o custo fixo de ~82 ms por rodada.

```bash
clang -O2 -dynamiclib -framework Metal -framework QuartzCore \
      -o gpuclock.dylib gpuclock.m
python3 test_gpuclock.py     # prova de vida: arma, roda, drena
```

API: `gpuclock_arm()` · `gpuclock_count()` · `gpuclock_drain(buf, n)` ·
`gpuclock_reset()` · `gpuclock_now()`.

Ressalva medida na pesquisa: atribuir por ORDEM de drenagem produz lixo (um
`gpu_span` maior que o relógio de parede). A leitura que vale usa o
`GPUStartTime` dentro da janela e drena com um ciclo de atraso.
