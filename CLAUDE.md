# CLAUDE.md

Servidor de inferência para Apple Silicon. Este repositório é um fork de
`jundot/omlx`, e o aplicativo que roda na barra superior do Mac é construído
daqui.

## A armadilha que mais custa tempo aqui

**O aplicativo traz a própria cópia do pacote `omlx` dentro de si.** Código
escrito neste repositório NÃO entra em vigor no servidor nem no painel web até
o aplicativo ser reconstruído.

```
o repositório        /Users/pedroberaldo/omlx-fork/omlx/
a cópia que roda     ~/Applications/oMLX Fork.app/Contents/Resources/omlx/
```

O ícone da barra, o servidor na porta 8000 e o painel em
`127.0.0.1:8000/admin` são o mesmo processo, e ele executa a cópia embutida.

Sintoma de esquecer isso: um conserto commitado parece não funcionar, e o erro
que ele resolve continua aparecendo no registro. Aconteceu em 31/08.

**Como reconstruir e instalar: [BUILD-DO-FORK.md](BUILD-DO-FORK.md).**

## Rotas

| assunto | onde |
|---|---|
| construir e instalar o aplicativo | [BUILD-DO-FORK.md](BUILD-DO-FORK.md) |
| medições, números e caminhos refutados | `~/PROGRAMACAO/oMLX-WORKS` |
| o que o projeto é, para quem chega | [README.md](README.md) |
| ajustar o teto de memória da placa | `~/Applications/Memória do Metal.app` ou `~/bin/omlx-iogpu.sh` |
| a sessão de 04/09 (performance) | `.claude/reports/2026-09-04-buffers-de-comando/README.md` |
| a noite de 04-05/09 (cabeça a 2 bits, aquecimento, contexto longo, PRs) | `.claude/reports/2026-09-04-cabeca-2-bits/README.md` |

A base de conhecimento em `~/PROGRAMACAO/oMLX-WORKS` guarda a tabela de números
medidos e a lista do que já foi refutado. **Consulte antes de citar número ou
propor caminho** — vários já foram testados e descartados.

## Regras de trabalho neste projeto

**Suba o servidor a partir da pasta pessoal, nunca de dentro do repositório.**
Com o diretório atual aqui, o pacote local vence o embutido e o ciclo de
importação derruba os núcleos Metal. O servidor avisa no registro e segue no
caminho lento:

```
native extension is present but failed to load; falling back to the slow path:
cannot import name '_ext' ... (most likely due to a circular import)
```

**Medição de velocidade exige conferir esse aviso antes.** Zero ocorrências é o
esperado; qualquer número medido com o aviso presente está no caminho lento e
não vale.

**Mas zero avisos não prova que os núcleos existem.** O aviso só sai quando o
arquivo está lá e falha ao carregar; quando ele não está, o registro fica limpo.
Dois builds do Fork saíram sem núcleo nenhum por isso. Confira por presença de
arquivo — [BUILD-DO-FORK.md](BUILD-DO-FORK.md), seção "Levar os núcleos Metal".

**Medição exige a máquina em repouso.** Confira a memória livre e os processos
antes de medir, não depois de estranhar o número. Calibração de quantização com
um modelo carregado no servidor já derrubou o servidor por falta de memória.

**Guardar a resposta crua é obrigatório ao avaliar qualidade.** A taxa de
acerto sozinha não distingue "errou a lógica" de "colapsou no meio", e essas
duas coisas dizem coisas opostas sobre o modelo.

**Amostra pequena não fecha conclusão.** Quatro afirmações caíram em 31/08 por
isso, três antes de sair do projeto. Quando o veredito depende de estatística,
dobre a amostra até duas rodadas seguidas repetirem o resultado.

**Enquanto uma quantização estiver rodando, nenhuma medição de GPU vale.** A regra acima diz
"máquina em repouso" e faltava dizer que a própria conversão conta como ocupação. Medido em
04/09: um `gather_qmm` de 2 bits deu 7,3 ms durante uma quantização, contra 0,108 ms em repouso —
68 vezes mais lento.

**Medir dois modelos exige derrubar e resubir o servidor entre eles.** O ledger do kernel
(`phys_footprint`) continua contando o modelo recém-descarregado, a folga de memória some, e o
guarda desce ao piso de 32 tokens por pedaço. Medido em 04/09: o modelo em produção deu 66-70
palavras por segundo medido logo após descarregar outro, contra 119,5 no protocolo limpo.

**Reiniciar o servidor quando o app da barra não responde.** A rota `POST /admin/api/server/restart`
só funciona sob o supervisor do app; matar o `omlx-server` não faz o app respawnar; e em 04/09 o
app travou de vez (AppleEvent −1712, ignorou SIGTERM, `~/.omlx/bin/omlx start` recusado no
socket de controle). O que funcionou: `cd ~ && ~/.omlx/bin/omlx serve` — o CLI do próprio app,
com o Python e o pacote embutidos, subiu em 3 s com os núcleos Metal e o teto de 121,6 GB.
Antes de derrubar, descarregue e desfixe o modelo (`PUT .../settings {"is_pinned": false}` e
`POST .../unload`), senão ele volta a carregar 103 GB sozinho.

**O `rm` é recusado pela permissão das sessões.** Para apagar modelo, use a rota do painel
(`DELETE /admin/api/hf/models/<nome>`), que também atualiza a lista do servidor. Para apps,
`osascript -e 'tell application "Finder" to delete ...'`.

## O teto de memória da placa

O servidor lê `iogpu.wired_limit_mb` **quando sobe**; é o teto físico (121,6 GB). O alvo em que o
guarda começa a encolher pedaços é uma fração dele — ver o `soft_threshold` abaixo.

Desde 04/09 o valor sobrevive ao reinício: há um serviço do sistema
(`/Library/LaunchDaemons/com.pedroberaldo.omlx-iogpu.plist`) que o reaplica no boot.

```
~/Applications/Memória do Metal.app   o app: mostra o estado e ajusta
~/bin/omlx-iogpu.sh [alto|padrao|fixar|soltar|status]
```

`fixar` grava o teto de agora para voltar sozinho; `status` diz se isso está no lugar. O
`launchctl bootstrap` falha com `Input/output error` quando já existe registro do mesmo serviço
— o `bootout` antes resolve, e o modo `fixar` já faz isso.

**O alvo do preparo (onde o guarda começa a encolher pedaços) vem do `soft_threshold` salvo no
`settings.json`, não do nível do guarda.** Em 04/09 estava em 0,90 (109,44 GB); em 05/09 passou a
0,925 (112,48 GB) com o nível `aggressive`, e a margem de aborto ficou em 0,95 (115,5 GB). Isso só
é seguro COM o aquecimento pós-carga ligado: sem ele, o 1º pedaço frio custa 14,5 GB e o pico
bate 117 GB (medido em 04/09). Perto do alvo, o estrangulador se autoalimenta (a leitura de
memória mede a mexida no pool e a taxa por token sobe em progressão geométrica) — `edf16963`
fecha um dos caminhos, mas a folga é o que evita o colapso. Detalhe em
`.claude/reports/2026-09-04-cabeca-2-bits/README.md`.

**Em contexto alto (>16k) o pedaço custa menos, não mais**: 285-412 MB contra ~1 GB abaixo. A
110k o bloco de 1024 rendeu 181 tok/s (94 pedaços inteiros, zero cortes) contra 163 com 512.

## O formulário de quantização tem quatro opções de memória

Desde `ac781500` (três) e `7fab3550` (a quarta), o formulário do painel expõe quatro escolhas,
todas com padrão que preserva o comportamento anterior:

| opção | valores | o que faz |
|---|---|---|
| teto de bits da atenção | 0 (sem teto) · 4 · 5 · 6 | a única regra que DESCE bits; sufixo `-a4` no nome |
| tamanho do grupo | 64 · 128 | quantos pesos dividem um fator de escala; sufixo `-g128` |
| dtype da torre de visão | auto · float16 · float32 | `auto` = float16 só em `glm5_next` |
| piso de bits da cabeça MTP | 4 (padrão) · 0 (acompanha o tronco) | solto, a cabeça de rascunho desce junto com o tronco; sufixo `-mtp2` |

**O preço das três primeiras está medido: ~0,03 de perplexidade por gigabyte economizado**,
constante nas duas receitas testadas. Não há receita esperta. Detalhe em
`.claude/reports/2026-09-04-buffers-de-comando/candidatos.md`.

**A quarta é a exceção, por construção:** a cabeça só escreve rascunhos e o tronco confere todo
token emitido, então a perplexidade sai idêntica e o que muda é a taxa de aceitação. Estimado
no checkpoint em 04/09: −1,81 GB (110,07 → 108,26 GB) no oQ2e. É a alavanca a puxar quando
falta folga ao preparo do prompt.

**Ajustar o bloco de preparo do modelo híbrido:** `OMLX_ARRAYS_CACHE_BLOCK=<n>` no ambiente do
servidor fixa o bloco do cache paginado E o `prefill_step_size` de uma vez
(`omlx/scheduler.py:2827-2840`); sem ele o servidor força 512 para o `glm5_next`.

## O aquecimento da placa (keepwarm)

Ligado de fábrica desde `14b44117`. A placa baixa o clock quando fica ociosa, e a primeira
geração depois de uma pausa paga a subida. Medido no servidor, duas rodadas por braço, com 8
segundos entre pedidos: **17,5-18,5 palavras por segundo sem ele contra 23,8-24,5 com ele**
(+29% a +40%), e zero custo em pedidos consecutivos.

O tique é uma multiplicação 256×256 em float16, disparada só no ramo do laço em que não há
requisição alguma (`omlx/engine_core.py`). Desliga em Settings › Advanced › Performance, ou com
`OMLX_GPU_KEEPWARM=0`.

## O aquecimento do preparo depois da carga (prefill warmup)

Ligado de fábrica desde `d5dc0a99`. O primeiro pedaço de prefill depois de uma carga fria custa
~14,5 GB de memória do processo (o pool de buffers do Metal enchendo uma vez); todos os seguintes,
~2 GB. Quem pagava era o primeiro prompt do usuário — medido em 04/09: o modelo de 103 GB cruzava a
marca dura de 115,5 GB e **o primeiro prompt era abortado em 2 de 4 cargas**; e o preditor do
estrangulador recusava o segundo pedaço de um bloco de 1024.

Agora o pool roda um pedaço de 512 tokens descartável logo depois do `Loaded model`, na thread MLX
do motor (`Scheduler.warmup_prefill`). Medido em 6 subidas: 5,9 s na carga, o guarda drena o pool
em 1 s (`hard -> ok`, 115,7 → 104,9 GB), e o primeiro prompt real passa a ver 2,4 GB de
transitório em vez de 14,5. Desliga em Settings › Advanced (`prefill_warmup`) ou com
`OMLX_PREFILL_WARMUP=0`.

**Ao medir prefill, o primeiro prompt de uma subida antiga (sem aquecimento) não é comparável ao
de uma subida nova** — era ele que carregava o custo frio.

## Onde ficam configuração e registros

```
~/.omlx/settings.json         chave da API, memória, amostragem
~/.omlx/model_settings.json   ajustes por modelo
~/.omlx/logs/server.log       o registro do servidor
~/.omlx/models/               os modelos em disco
```

O servidor reescreve `model_settings.json` ao subir, removendo entradas de
modelos que não estão mais em disco.

## Testes

```bash
.venv/bin/python3 -m pytest tests/ -q                    # a suíte inteira
.venv/bin/python3 -m pytest tests/ -q -k "glm or moe"    # um recorte
```

O ambiente é gerido por `uv` e não tem `pip`: instalar exige `VIRTUAL_ENV`
setado e `uv pip install`.

**Nove falhas da suíte são anteriores e não são regressão** (conferidas em 31/08
rodando no commit anterior ao merge do tronco). Qualquer falha ALÉM destas é
regressão e trava a entrega:

```
3  test_cluster_performance.py          (sampling rank · vocab projection · staggered prompt)
3  test_cluster_progressive_loading.py  (native tensor strategy)
2  test_mlx_vlm_qwen4_exp_compat.py     hyper_connection_fails_closed[False]/[True]
1  test_glm_moe_dsa_patch.py            glm_native_fused_kernels_match_reference
```

A última **não é defeito**: ela exige `== 0.0` entre o núcleo Metal fundido e a
referência, e a diferença medida é 1,2e-4 em fp16 — ruído de arredondamento de uma
soma feita em ordem diferente. Fica listada porque o teste é frágil, não o núcleo.

O `test_plan_helper_classification` **saiu desta lista**: era doc e teste
desatualizados depois que `PoolingCache` entrou de propósito no caminho por membro
para o GLM-5.x, e os três (código, descrição e teste) foram alinhados em 31/08.

## Commits

O dono é o único autor. **Nunca** adicione `Co-Authored-By` nem qualquer linha
mencionando Claude ou Anthropic nas mensagens de commit.
