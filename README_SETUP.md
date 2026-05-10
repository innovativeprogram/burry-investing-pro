# Configurazione Sicura di V-QUANT PRO

## Integrazione Supabase
Per far funzionare il progetto correttamente senza esporre dati sensibili:

1. Crea un file `.env` locale basato su questo schema:
   ```
   SUPABASE_URL=tua_url_progetto
   SUPABASE_ANON_KEY=tua_chiave_anonima
   ```
2. **MAI** inserire la `SERVICE_ROLE_KEY` nel file `.env` se quest'ultimo viene caricato in un ambiente client-side.
3. Su GitHub, configura i 'Secrets' per le tue pipeline di CI/CD.