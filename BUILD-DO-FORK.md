# Como o oMLX Fork.app é construído a partir deste repositório

O aplicativo que roda na barra superior do Mac é **construído deste repositório**.
Ele traz a própria cópia do pacote `omlx` dentro de si, em
`Contents/Resources/omlx/`. Código escrito aqui **não entra em vigor** no painel
web nem no servidor até o aplicativo ser reconstruído.

Isso custou tempo em 31/08: um adaptador commitado no repositório parecia
quebrado porque o servidor lia a cópia antiga de dentro do aplicativo.

## O que é o quê

| coisa | onde vive | quem usa |
|---|---|---|
| o código-fonte | `/Users/pedroberaldo/omlx-fork/omlx/` | scripts que você roda à mão |
| o aplicativo | `~/Applications/oMLX Fork.app` | o ícone da barra, o servidor e o painel |
| a cópia embutida | `oMLX Fork.app/Contents/Resources/omlx/` | é ELA que o servidor executa |

O ícone da barra, o servidor na porta 8000 e o painel em `127.0.0.1:8000/admin`
são o mesmo processo, servido pela cópia embutida.

## Construir

```bash
cd /Users/pedroberaldo/omlx-fork

# 1. o build oficial. O app atual serve de doador do ambiente Python,
#    o que dispensa remontar as camadas do zero (que leva muito mais tempo).
OMLX_DONOR_APP="$HOME/Applications/oMLX Fork.app" \
  bash apps/omlx-mac/Scripts/build.sh release --no-rebuild-donor

# sai em: apps/omlx-mac/build/Stage/oMLX.app   (~90 s, exige Xcode)
```

O `xcodebuild` fixa o nome `oMLX`. A identidade do Fork é aplicada depois:

```bash
cd apps/omlx-mac/build/Stage
ditto oMLX.app "oMLX Fork.app"
P="oMLX Fork.app/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleName 'oMLX Fork'"        "$P"
/usr/libexec/PlistBuddy -c "Set :CFBundleDisplayName 'oMLX Fork'" "$P"
/usr/libexec/PlistBuddy -c "Set :CFBundleIdentifier studio.viu.omlx-fork" "$P"
codesign --force --deep --sign - "oMLX Fork.app"
```

`studio.viu.omlx-fork` **não está em nenhum arquivo versionado** — sem esses três
ajustes o build sai como `oMLX` / `app.omlx`, que o macOS trata como outro
aplicativo.

## Instalar

```bash
# derruba o servidor e o app
for P in $(lsof -nP -iTCP:8000 -sTCP:LISTEN -t); do kill "$P"; done
pkill -f "oMLX Fork.app/Contents/MacOS/oMLX"

# guarda o anterior — sempre, porque é o ambiente de trabalho
mv ~/Applications/oMLX\ Fork.app ~/Applications/oMLX\ Fork.app.<versao>-backup

ditto "apps/omlx-mac/build/Stage/oMLX Fork.app" ~/Applications/oMLX\ Fork.app
```

## Levar os núcleos Metal — o build NÃO faz isso sozinho

O empacotamento sai com a árvore `custom_kernels/` presente e **vazia de binários**.
Sem eles o servidor sobe, carrega modelo e gera texto, então nada parece errado —
mas a conta de memória do preparo passa a cobrar a faixa mais cara e o guarda
recusa prompts de qualquer tamanho.

A fonte certa é o **instalador da release, da sua versão E do seu macOS**. A
release publica dois, e eles não são intercambiáveis:

```
oMLX-<versão>-macos15-sequoia.dmg     para macOS 15
oMLX-<versão>-macos26-27.dmg          para macOS 26 e 27
```

Copiar é seguro: o Fork nunca alterou essa pasta
(`git log jundot/main..HEAD -- omlx/custom_kernels/` sai vazio).

```bash
V=$(python3 -c "import re,pathlib; \
  print(re.search(r'[0-9]+\.[0-9]+\.[0-9]+[a-z0-9]*', \
  pathlib.Path('omlx/_version.py').read_text()).group())")
OS=$(sw_vers -productVersion | cut -d. -f1)
case "$OS" in 15) SUF="macos15-sequoia" ;; 26|27) SUF="macos26-27" ;; \
  *) echo "macOS $OS não previsto — confira os anexos da release"; exit 1 ;; esac

FORK=~/Applications/oMLX\ Fork.app/Contents/Resources/omlx/custom_kernels
T=$(mktemp -d)

gh release download "v$V" --repo jundot/omlx \
  --pattern "oMLX-$V-$SUF.dmg" --dir "$T"
hdiutil attach "$T/oMLX-$V-$SUF.dmg" -nobrowse -readonly -mountpoint "$T/mnt"
SRC="$T/mnt/oMLX.app/Contents/Resources/omlx/custom_kernels"

for d in bonsai decode_fast glm_moe_dsa minimax_m3 qwen35_prefill; do
  mkdir -p "$FORK/$d"
  cp -p "$SRC/$d"/*.so "$SRC/$d"/*.dylib "$SRC/$d"/*.metallib "$FORK/$d/" 2>/dev/null
done

hdiutil detach "$T/mnt"
open -a ~/Applications/oMLX\ Fork.app
```

**As rodas do PyPI NÃO servem em macOS 26.** As três publicadas
(`omlx-<versão>-cp31X-cp31X-macosx_15_0_universal2.whl`) são todas de piso macOS
15, e nenhum dos dezesseis binários bate com o do instalador de macOS 26 — os
quinze comuns têm conteúdo diferente, e o décimo sexto,
`omlx_qwen35_prefill_kernels_nax.metallib`, só existe quando o build usa o SDK
26.2 (`csrc/CMakeLists.txt:125`). Ele é o suporte ao acelerador neural.

**E não copie de um aplicativo instalado.** O oMLX de `/Applications` desta
máquina está parado na 0.6.3rc3; copiar dali entrega binários velhos que carregam
sem erro e simplesmente não têm os símbolos novos — três do Qwen4 faltavam, e o
código apenas cai no caminho lento, em silêncio.

Compilar do próprio repositório continua sendo a saída para código de núcleo que
o Fork venha a alterar. Exige apontar o interpretador do aplicativo: o ambiente
daqui usa Python 3.12 e o aplicativo roda 3.11, então
`_ext.cpython-312-darwin.so` não carrega dentro dele. O `setup.py` aceita
`Python_EXECUTABLE` via `CMAKE_ARGS` (setup.py:46-47).

## Conferir que deu certo

```bash
# a versão subiu
plutil -p ~/Applications/oMLX\ Fork.app/Contents/Info.plist | grep CFBundleVersion

# o código novo está lá dentro
grep -A 1 "_STREAM_CALIBRATION_SUPPORTED_MODEL_TYPES" \
  ~/Applications/oMLX\ Fork.app/Contents/Resources/omlx/oq.py

# os núcleos Metal ESTÃO no pacote (13 é o esperado; zero passou despercebido
# em dois builds seguidos)
find ~/Applications/oMLX\ Fork.app/Contents/Resources/omlx \
     \( -name "*.so" -o -name "*.dylib" -o -name "*.metallib" \) | wc -l
# 16 na 0.6.4 em macOS 26; 15 em macOS 15 (lá não há o binário do acelerador
# neural). Contar não basta: um binário de outra versão ou de outra plataforma
# também conta, e é por isso que a conferência de símbolo abaixo existe.

# e RESPONDEM — de fora do repositório, senão você testa a cópia local
APP=~/Applications/oMLX\ Fork.app/Contents/Resources
env -C "$HOME" \
  PYTHONPATH="$APP:$APP/Python/framework-mlx-base/lib/python3.11/site-packages" \
  "$APP/Python/cpython-3.11/bin/python3" -c \
  "from omlx.custom_kernels.glm_moe_dsa import fast as f; \
   print(f.has_symbol('glm_dsa_sparse_mla_attention'))"      # espera-se True

# nenhum aviso de caminho lento (zero é o esperado, mas ver a armadilha abaixo)
grep -c "native extension is present but failed" ~/.omlx/logs/server.log
```

## Armadilhas medidas

**Não suba o servidor com o diretório atual dentro do repositório.** O pacote
local vence o embutido e o ciclo de importação derruba os núcleos Metal:

```
native extension is present but failed to load; falling back to the slow path:
cannot import name '_ext' ... (most likely due to a circular import)
```

Sem os núcleos, o transiente do preparo cresce e o guarda de memória recusa
prompts que caberiam. Suba sempre a partir da pasta pessoal.

**Contar avisos no registro NÃO prova que os núcleos estão lá.** O aviso só sai
quando o arquivo existe e falha ao carregar. Quando ele não existe, a checagem é
um `except` silencioso e o registro fica limpo — indistinguível de tudo certo.
Foi assim que dois builds do Fork saíram sem núcleo nenhum sem ninguém notar, até
o guarda de memória recusar um prompt de 135 palavras em 31/08:

```
Prefill would require ~117.13 GB peak (current 103.67 GB + KV+SDPA 13.46 GB)
but dynamic ceiling is 116.06 GB.
```

Com os núcleos no lugar, o mesmo prompt passou e o registro não trouxe nenhuma
recusa. **Confira por presença de arquivo e por resposta de símbolo, nunca por
ausência de aviso.**

**Toda medição de velocidade feita com esse aviso no registro está no caminho
lento e não vale.** Confira o registro antes de citar qualquer número.

**O modo `swift` do build exige um export que pode não existir.** Se
`packaging/_export/` estiver ausente, `build.sh swift` recusa; use
`release --no-rebuild-donor` com o app atual como doador.

## Onde ficam configuração e registros

```
~/.omlx/settings.json         chave da API, memória, amostragem
~/.omlx/model_settings.json   ajustes por modelo (rascunhador, cache)
~/.omlx/logs/server.log       o registro do servidor
~/.omlx/models/               os modelos em disco
```

O servidor reescreve `model_settings.json` ao subir, e entradas de modelos que
não estão mais em disco são removidas.
