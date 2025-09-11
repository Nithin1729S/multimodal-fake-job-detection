import random
import pandas as pd
import os
import time
import json
import hashlib
import sys
import urllib.parse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

# =========================
# Configurations
# =========================
OUTPUT_DIR = "images"
CACHE_FILE = "company_cache.json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parse command line arguments
start_job_id = None
if len(sys.argv) > 1:
    try:
        start_job_id = int(sys.argv[1])
        print(f"[INFO] Starting from job_id: {start_job_id}")
    except ValueError:
        print(f"[ERROR] Invalid job_id provided: {sys.argv[1]}. Must be an integer.")
        sys.exit(1)

# Load or create cache
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

def get_cache_key(company_profile):
    # Create a hash of the company profile for caching
    return hashlib.md5(company_profile.encode()).hexdigest()[:10]

# Load dataset and take first 10 samples, limit company_profile to 50 chars
df = pd.read_csv("fake_job_postings.csv")

# Replace missing or empty company_profile with description
df['company_profile'] = df['company_profile'].fillna('')
df['description'] = df['description'].fillna('')

df['company_profile'] = df.apply(
    lambda row: row['company_profile'] if row['company_profile'].strip() != '' else row['description'],
    axis=1
)

# Limit company_profile to first 50 characters
df['company_profile'] = df['company_profile'].str[:50]

# Filter dataset based on start_job_id if provided
if start_job_id is not None:
    start_index = df[df['job_id'] == start_job_id].index
    if len(start_index) == 0:
        print(f"[ERROR] Job ID {start_job_id} not found in dataset.")
        sys.exit(1)
    df = df.loc[start_index[0]:].reset_index(drop=True)
    print(f"[INFO] Dataset filtered to start from job_id {start_job_id}. {len(df)} rows remaining.")

# Load existing cache
company_cache = load_cache()

# Setup Selenium WebDriver with full desktop settings
options = webdriver.ChromeOptions()
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

# Remove headless mode and set window size for full desktop experience
options.add_argument("--start-maximized")
options.add_argument("--window-size=1920,1080")
options.add_argument("--force-device-scale-factor=1")
options.add_argument("--disable-web-security")
options.add_argument("--allow-running-insecure-content")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

# Set window size explicitly after driver initialization
driver.set_window_size(1920, 1080)
driver.maximize_window()

# =========================
# Helper functions
# =========================
def accept_cookies():
    """Try to accept cookies on the current page"""
    cookie_selectors = [
        # Common cookie acceptance button selectors
        "button[id*='accept']",
        "button[class*='accept']",
        "button:contains('Accept all')",
        "button:contains('Accept All')",
        "button:contains('ACCEPT ALL')",
        "button:contains('Accept')",
        "button:contains('OK')",
        "button:contains('I agree')",
        "button:contains('Agree')",
        "button:contains('Got it')",
        "a[id*='accept']",
        "a[class*='accept']",
        "[data-testid*='accept']",
        "[id*='cookie'] button",
        "[class*='cookie'] button",
        ".cookie-banner button",
        "#cookie-banner button",
        ".gdpr-banner button",
        "#gdpr-banner button",
        ".consent button",
        "#consent button"
    ]
    
    for selector in cookie_selectors:
        try:
            # Try CSS selector first
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            for element in elements:
                if element.is_displayed() and element.is_enabled():
                    try:
                        element.click()
                        print(f"[INFO] Clicked cookie button with selector: {selector}")
                        # time.sleep(2)
                        return True
                    except:
                        continue
        except:
            continue
    
    # Try XPath for text-based selections
    text_selectors = [
        "//button[contains(text(), 'Accept')]",
        "//button[contains(text(), 'OK')]", 
        "//button[contains(text(), 'I agree')]",
        "//button[contains(text(), 'Got it')]",
        "//a[contains(text(), 'Accept')]",
        "//div[contains(@class, 'cookie')]//button",
        "//div[contains(@class, 'consent')]//button"
    ]
    
    for xpath in text_selectors:
        try:
            elements = driver.find_elements(By.XPATH, xpath)
            for element in elements:
                if element.is_displayed() and element.is_enabled():
                    try:
                        element.click()
                        print(f"[INFO] Clicked cookie button with xpath: {xpath}")
                        # time.sleep(2)
                        return True
                    except:
                        continue
        except:
            continue
    
    return False

def is_linkedin_page(url):
    """Check if the URL is a LinkedIn page"""
    return 'linkedin.com' in url.lower()

def capture_company_image(job_id, company_profile, label):
    try:
        print(f"[INFO] Processing job_id={job_id}, company='{company_profile}'")
        
        # Check cache first
        cache_key = get_cache_key(company_profile)
        if cache_key in company_cache:
            cached_path = company_cache[cache_key]
            if os.path.exists(cached_path):
                print(f"[CACHE] Using cached image for company: {company_profile}")
                # Copy cached image to current job_id
                label_folder = os.path.join(OUTPUT_DIR, str(label))
                os.makedirs(label_folder, exist_ok=True)
                new_path = os.path.join(label_folder, f"{job_id}.png")
                
                import shutil
                shutil.copy2(cached_path, new_path)
                return new_path 
        # else:
        #     return None
        # Create search query and encode it for URL
        # time.sleep(random.uniform(2, 5)  )
        search_query = f"{company_profile} company profile"
        encoded_query = urllib.parse.quote_plus(search_query)
        
        # Navigate directly to DuckDuckGo search results using URL parameters
        duckduckgo_search_url = f"https://duckduckgo.com/?q={encoded_query}"
        driver.get(duckduckgo_search_url)
        
        print(f"[INFO] Searching for: '{search_query}' via URL: {duckduckgo_search_url}")
        
        # Wait for search results to load - DuckDuckGo uses different selectors
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='result']"))
        )
        # time.sleep(3)
        
        # Accept cookies on DuckDuckGo
        accept_cookies()
        
        # DuckDuckGo specific selectors for search results
        result_selectors = [
            "[data-testid='result'] h2 a[href]",
            "[data-testid='result'] a[href]:not([href*='duckduckgo.com'])",
            "article h2 a[href]",
            ".result__a[href]",
            "a[href*='http']:not([href*='duckduckgo.com'])",
        ]
        
        first_link = None
        attempts = 0
        max_attempts = 10  # Try up to 10 results to avoid LinkedIn
        
        for selector in result_selectors:
            try:
                results = driver.find_elements(By.CSS_SELECTOR, selector)
                for result in results[:max_attempts]:  # Limit attempts
                    href = result.get_attribute("href")
                    if (href and 
                        not href.startswith("https://duckduckgo.com") and 
                        not href.startswith("http://duckduckgo.com") and
                        "duckduckgo.com" not in href and
                        not is_linkedin_page(href)):  # Skip LinkedIn pages
                        
                        first_link = href
                        print(f"[INFO] Found non-LinkedIn link using selector: {selector}")
                        print(f"[INFO] Link: {href}")
                        break
                    elif is_linkedin_page(href):
                        print(f"[INFO] Skipping LinkedIn page: {href}")
                        attempts += 1
                        
                if first_link:
                    break
            except Exception as e:
                continue
        
        if not first_link:
            print(f"[WARN] No non-LinkedIn results found for job_id={job_id}")
            return None
        
        print(f"[INFO] Navigating to: {first_link}")
        
        # Navigate to first result
        driver.get(first_link)
        # time.sleep(7)
        
        # Accept cookies on the target website
        accept_cookies()
        
        # Scroll to ensure full page is rendered
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        # time.sleep(2)
        driver.execute_script("window.scrollTo(0, 0);")
        # time.sleep(2)
        
        # Get page dimensions for full screenshot
        page_width = driver.execute_script("return Math.max(document.body.scrollWidth, document.documentElement.scrollWidth, document.body.offsetWidth, document.documentElement.offsetWidth, document.body.clientWidth, document.documentElement.clientWidth);")
        page_height = driver.execute_script("return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight, document.body.offsetHeight, document.documentElement.offsetHeight, document.body.clientHeight, document.documentElement.clientHeight);")
        
        print(f"[INFO] Page dimensions: {page_width}x{page_height}")
        
        # Temporarily set window size to capture full page
        driver.set_window_size(max(page_width, 1920), max(page_height, 1080))
        # time.sleep(2)
        
        # Create label folder
        label_folder = os.path.join(OUTPUT_DIR, str(label))
        os.makedirs(label_folder, exist_ok=True)
        
        # Save full desktop-sized screenshot
        image_path = os.path.join(label_folder, f"{job_id}.png")
        
        # Take screenshot of full page
        driver.save_screenshot(image_path)
        
        # Cache the image path
        company_cache[cache_key] = image_path
        save_cache(company_cache)
        
        # Reset window size back to standard size for headless mode
        driver.set_window_size(1920, 1080)
        
        print(f"[SUCCESS] Full desktop screenshot saved and cached: {image_path}")
        return image_path

    except Exception as e:
        print(f"[ERROR] job_id={job_id} - {e}")
        return None

# =========================
# Main loop
# =========================
try:
    print(f"[INFO] Starting with window size: {driver.get_window_size()}")
    print(f"[INFO] Cache contains {len(company_cache)} entries")
    
    if start_job_id is not None:
        print(f"[INFO] Processing {len(df)} rows starting from job_id {start_job_id}")
    else:
        print(f"[INFO] Processing all {len(df)} rows")
    
    image_paths = []
    for idx, row in df.iterrows():
        print(f"[INFO] Processing row {idx + 1}/{len(df)} (job_id={row['job_id']})")
        img_path = capture_company_image(row['job_id'], row['company_profile'], row['fraudulent'])
        image_paths.append(img_path)
        print(f"[INFO] Completed job_id={row['job_id']}")
        # time.sleep(random.uniform(2, 5)  )  # Be respectful between requests
    
    # Update dataframe and save
    df['image_path'] = image_paths
    
    # Create output filename based on whether we started from a specific job_id
    if start_job_id is not None:
        output_filename = f"fake_job_postings_with_images_from_{start_job_id}.csv"
    else:
        output_filename = "fake_job_postings_with_images.csv"
    
    df.to_csv(output_filename, index=False)
    
    print(f"Dataset updated with full desktop screenshot paths! Saved as: {output_filename}")
    print(f"Final cache contains {len(company_cache)} entries")
    
finally:
    driver.quit()