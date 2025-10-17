Database initialization

- Place your full MySQL dump into `data/casaos.sql` (or import it directly into your server).
- The schema provided in your request is compatible with this backend. If you want a ready-to-run import file, copy the dump you pasted and save it as `data/casaos.sql`.

Quick import example (MySQL/MariaDB):

1. Create database and user if needed.
2. Import: `mysql -u <user> -p < database_name < data/casaos.sql`

Environment variables expected by the app are in `.env` and `app/core/config.py`.

