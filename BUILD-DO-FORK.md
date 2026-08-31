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
| o código-fonte | `/Users/pedroberaldo/omlx-qwen38-flash/omlx/` | scripts que você roda à mão |
| o aplicativo | `~/Applications/oMLX Fork.app` | o ícone da barra, o servidor e o painel |
| a cópia embutida | `oMLX Fork.app/Contents/Resources/omlx/` | é ELA que o servidor executa |

O ícone da barra, o servidor na porta 8000 e o painel em `127.0.0.1:8000/admin`
são o mesmo processo, servido pela cópia embutida.

## Construir

```bash
cd /Users/pedroberaldo/omlx-qwen38-flash

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
open -a ~/Applications/oMLX\ Fork.app
```

## Conferir que deu certo

```bash
# a versão subiu
plutil -p ~/Applications/oMLX\ Fork.app/Contents/Info.plist | grep CFBundleVersion

# o código novo está lá dentro
grep -A 1 "_STREAM_CALIBRATION_SUPPORTED_MODEL_TYPES" \
  ~/Applications/oMLX\ Fork.app/Contents/Resources/omlx/oq.py

# os núcleos Metal carregaram (zero é o esperado)
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
