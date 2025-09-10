import pandas as pd
import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# =========================
# Configurations
# =========================
OUTPUT_DIR = "images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset and take first 5 samples
df = pd.read_csv("fake_job_postings.csv")
df = df.dropna(subset=['company_profile']).head(10)

# Setup Selenium WebDriver with full desktop settings
options = webdriver.ChromeOptions()
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

# Remove headless mode and set window size for full desktop experience
# options.add_argument("--headless")  # Commented out for full desktop view
options.add_argument("--start-maximized")  # Start with maximized window
options.add_argument("--window-size=1920,1080")  # Set specific desktop resolution
options.add_argument("--force-device-scale-factor=1")  # Ensure proper scaling
options.add_argument("--disable-web-security")  # Help with some sites
options.add_argument("--allow-running-insecure-content")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

# Set window size explicitly after driver initialization
driver.set_window_size(1920, 1080)
driver.maximize_window()

# =========================
# Helper function
# =========================
def capture_company_image(job_id, company_profile, label):
    try:
        print(f"[INFO] Processing job_id={job_id}, company='{company_profile}'")
        
        # Navigate to Google
        driver.get("https://www.google.com/")
        time.sleep(3)
        
        # Find and use search box
        search_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.NAME, "q"))
        )
        search_query = f"{company_profile} company profile"
        search_box.clear()
        search_box.send_keys(search_query)
        search_box.send_keys(Keys.RETURN)
        
        print(f"[INFO] Searching for: '{search_query}'")
        
        # Wait for search results to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "search"))
        )
        time.sleep(3)
        
        # Try multiple selectors for search results
        result_selectors = [
            "div#search div.g a[href]:not([href*='google.com'])",  # Standard results
            "div.g > div > div > a[href]",  # Alternative structure
            "h3 a[href]",  # Direct h3 links
            "a[href*='http']:not([href*='google.com'])",  # Any external link
        ]
        
        first_link = None
        for selector in result_selectors:
            try:
                results = driver.find_elements(By.CSS_SELECTOR, selector)
                # Filter out Google's own links
                for result in results:
                    href = result.get_attribute("href")
                    if (href and 
                        not href.startswith("https://www.google.") and 
                        not href.startswith("https://google.") and
                        not href.startswith("http://www.google.") and
                        not href.startswith("http://google.") and
                        not href.startswith("https://support.google.") and
                        not href.startswith("https://accounts.google.") and
                        "google.com" not in href):
                        first_link = href
                        print(f"[INFO] Found link using selector: {selector}")
                        print(f"[INFO] Link: {href}")
                        break
                if first_link:
                    break
            except Exception as e:
                continue
        
        if not first_link:
            print(f"[WARN] No results found for job_id={job_id}")
            return None
        
        print(f"[INFO] Navigating to: {first_link}")
        
        # Navigate to first result
        driver.get(first_link)
        time.sleep(7)  # Give page time to load completely
        
        # Scroll to ensure full page is rendered
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(2)
        
        # Get page dimensions for full screenshot
        page_width = driver.execute_script("return Math.max(document.body.scrollWidth, document.documentElement.scrollWidth, document.body.offsetWidth, document.documentElement.offsetWidth, document.body.clientWidth, document.documentElement.clientWidth);")
        page_height = driver.execute_script("return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight, document.body.offsetHeight, document.documentElement.offsetHeight, document.body.clientHeight, document.documentElement.clientHeight);")
        
        print(f"[INFO] Page dimensions: {page_width}x{page_height}")
        
        # Temporarily set window size to capture full page
        driver.set_window_size(max(page_width, 1920), max(page_height, 1080))
        time.sleep(2)
        
        # Create label folder
        label_folder = os.path.join(OUTPUT_DIR, str(label))
        os.makedirs(label_folder, exist_ok=True)
        
        # Save full desktop-sized screenshot
        image_path = os.path.join(label_folder, f"{job_id}.png")
        
        # Take screenshot of full page
        driver.save_screenshot(image_path)
        
        # Reset window size back to desktop
        driver.set_window_size(1920, 1080)
        driver.maximize_window()
        
        print(f"[SUCCESS] Full desktop screenshot saved: {image_path}")
        return image_path

    except Exception as e:
        print(f"[ERROR] job_id={job_id} - {e}")
        return None

# =========================
# Main loop (first 5 only)
# =========================
try:
    print(f"[INFO] Starting with window size: {driver.get_window_size()}")
    
    image_paths = []
    for idx, row in df.iterrows():
        print(f"[INFO] Processing row {idx + 1}/5")
        img_path = capture_company_image(row['job_id'], row['company_profile'], row['fraudulent'])
        image_paths.append(img_path)
        print(f"[INFO] Completed job_id={row['job_id']}")
        time.sleep(3)  # Be respectful to Google and allow time between requests
    
    # Update dataframe and save
    df['image_path'] = image_paths
    df.to_csv("fake_job_postings_with_images.csv", index=False)
    
    print("Dataset updated with full desktop screenshot paths for first 5 samples!")
    
finally:
    driver.quit()