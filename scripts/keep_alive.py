import asyncio
from playwright.async_api import async_playwright

async def wake_up_app(url):
    print(f"🌐 Visitando {url}...")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            # Carica la pagina
            response = await page.goto(url, wait_until="domcontentloaded", timeout=120_000)
            
            # Verifica se la pagina è stata caricata correttamente
            if response and response.status >= 400:
                print(f"⚠️ La pagina ha risposto con stato {response.status}")
            
            await page.wait_for_timeout(5000)  # Aspetta 5 secondi

            # Cerca il bottone di risveglio
            wake_button = page.get_by_role("button", name="Yes, get this app back up!")

            if await wake_button.count() > 0:
                print(f"🔘 Trovato bottone di risveglio per {url}")
                await wake_button.click()
                await page.wait_for_timeout(60_000)  # Aspetta 60 secondi
                print(f"✅ App risvegliata: {url}")
            else:
                # Controlla se siamo sulla pagina attiva
                title = await page.title()
                print(f"🟢 App già attiva: {url} (Titolo: {title})")

        except Exception as e:
            print(f"❌ Errore con {url}: {e}")
        finally:
            await browser.close()

async def main():
    url = "https://vquantpro.streamlit.app/"
    print(f"🚀 Avvio script Keep-Alive per Streamlit")
    await wake_up_app(url)
    print("🏁 Script completato")

if __name__ == "__main__":
    asyncio.run(main())