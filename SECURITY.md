# Politica di Sicurezza

## Segnalazione di Vulnerabilità

La sicurezza dei dati dei nostri utenti è la nostra massima priorità. Se ritieni di aver trovato una vulnerabilità di sicurezza in questo progetto, ti preghiamo di non aprire una 'Issue' pubblica.

Invia invece una segnalazione dettagliata a: **security@tuodominio.it** (sostituire con la propria email).

## Gestione dei Dati (Supabase)
Tutti i dati sensibili relativi agli account sono gestiti tramite l'infrastruttura di Supabase. 
- Utilizziamo la **Row Level Security (RLS)** per garantire che ogni utente possa accedere solo ai propri dati.
- Le chiavi 'Service Role' non sono mai esposte nel codice sorgente.