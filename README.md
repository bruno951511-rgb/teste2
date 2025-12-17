# Terminal AI V2 - Advanced Edition

## \ud83d\ude80 Recursos

- \u2705 Execu\u00e7\u00e3o REAL de comandos Linux
- \u2705 Cria\u00e7\u00e3o REAL de arquivos (HTML, Python, JS, etc)
- \u2705 Sistema de m\u00faltiplas chaves Gemini com rota\u00e7\u00e3o autom\u00e1tica
- \u2705 Upload e an\u00e1lise de imagens
- \u2705 Interface avan\u00e7ada com c\u00f3digo copi\u00e1vel
- \u2705 Hist\u00f3rico de chats no MongoDB
- \u2705 Sistema de mem\u00f3ria persistente

## \ud83d\udee0\ufe0f Configura\u00e7\u00e3o

1. Edite o arquivo `.env` e adicione suas chaves Gemini:
   ```
   GEMINI_KEYS=chave1,chave2,chave3,chave4,chave5
   ```

2. Inicie o servidor:
   ```bash
   ./start.sh
   ```

3. Abra o frontend: `terminal_ai_v2_advanced.html`

4. Configure o ngrok (opcional):
   ```bash
   ngrok http 8001
   ```

## \ud83d\udcca Estat\u00edsticas

- Com 5 chaves: 100 requisi\u00e7\u00f5es/dia
- Com 10 chaves: 200 requisi\u00e7\u00f5es/dia
- Com 20 chaves: 400 requisi\u00e7\u00f5es/dia

## \ud83d\udcda Comandos

```bash
./start.sh          # Iniciar servidor
./test.sh           # Testar instala\u00e7\u00e3o
curl localhost:8001/api/health  # Verificar status
```

## \ud83d\udc1b Bugs Corrigidos

- \u2713 JSON n\u00e3o aparece mais nas mensagens
- \u2713 Arquivos s\u00e3o criados de verdade
- \u2713 Loading desaparece corretamente
- \u2713 Scroll autom\u00e1tico funciona
- \u2713 Mensagens fantasmas removidas

## \ud83c\udf89 Vers\u00e3o 2.0

Desenvolvido com IA avan\u00e7ada para programa\u00e7\u00e3o!
