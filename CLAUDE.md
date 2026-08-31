# CLAUDE.md

Servidor de inferência para Apple Silicon. Este repositório é um fork de
`jundot/omlx`, e o aplicativo que roda na barra superior do Mac é construído
daqui.

## A armadilha que mais custa tempo aqui

**O aplicativo traz a própria cópia do pacote `omlx` dentro de si.** Código
escrito neste repositório NÃO entra em vigor no servidor nem no painel web até
o aplicativo ser reconstruído.

```
o repositório        /Users/pedroberaldo/omlx-qwen38-flash/omlx/
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

Duas falhas da suíte são anteriores a este trabalho e não são regressão:
`test_qwen4_small_hyper_connection_fusion_fails_closed` (falha no tronco puro) e
`test_plan_helper_classification` (um commit de terceiro mudou a regra e não
atualizou o teste).

## Commits

O dono é o único autor. **Nunca** adicione `Co-Authored-By` nem qualquer linha
mencionando Claude ou Anthropic nas mensagens de commit.
