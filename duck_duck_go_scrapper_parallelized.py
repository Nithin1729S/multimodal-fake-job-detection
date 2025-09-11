import random
import pandas as pd
import os
import time
import json
import hashlib
import sys
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import threading
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

# Thread-safe locks
cache_lock = Lock()
file_lock = Lock()

# Number of parallel workers (adjust based on your system)
MAX_WORKERS = 12  # Start with 4, can increase to 8-12 if system handles it

def create_driver():
    """Create a new WebDriver instance for each thread"""
    options = webdriver.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    # Add headless mode for better performance
    options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--force-device-scale-factor=1")
    options.add_argument("--disable-web-security")
    options.add_argument("--allow-running-insecure-content")
    
    # Performance optimizations
    options.add_argument("--disable-images")  # Don't load images for faster page loads
    options.add_argument("--disable-javascript")  # Disable JS if not needed
    options.add_argument("--disable-plugins")
    options.add_argument("--disable-extensions")
    options.add_argument("--no-first-run")
    options.add_argument("--disable-background-timer-throttling")
    options.add_argument("--disable-renderer-backgrounding")
    options.add_argument("--disable-backgrounding-occluded-windows")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    driver.set_window_size(1920, 1080)
    
    return driver

# Load or create cache
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with cache_lock:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)

def get_cache_key(company_profile):
    return hashlib.md5(company_profile.encode()).hexdigest()[:10]

def accept_cookies(driver):
    """Try to accept cookies on the current page"""
    cookie_selectors = [
        "button[id*='accept']",
        "button[class*='accept']",
        "button:contains('Accept')",
        "[data-testid*='accept']",
        ".cookie-banner button",
        "#cookie-banner button"
    ]
    
    for selector in cookie_selectors[:3]:  # Limit to most common selectors
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            for element in elements:
                if element.is_displayed() and element.is_enabled():
                    try:
                        element.click()
                        return True
                    except:
                        continue
        except:
            continue
    return False

def is_linkedin_page(url):
    return 'linkedin.com' in url.lower()

def process_single_job(job_data, company_cache):
    """Process a single job - this will run in parallel"""
    job_id, company_profile, label = job_data
    driver = None
    
    try:
        print(f"[INFO] Thread {threading.current_thread().name}: Processing job_id={job_id}")
        
        # Check cache first (thread-safe)
        cache_key = get_cache_key(company_profile)
        with cache_lock:
            if cache_key in company_cache:
                cached_path = company_cache[cache_key]
                if os.path.exists(cached_path):
                    print(f"[CACHE] Thread {threading.current_thread().name}: Using cached image for job_id={job_id}")
                    # Copy cached image to current job_id
                    label_folder = os.path.join(OUTPUT_DIR, str(label))
                    os.makedirs(label_folder, exist_ok=True)
                    new_path = os.path.join(label_folder, f"{job_id}.png")
                    
                    import shutil
                    shutil.copy2(cached_path, new_path)
                    return job_id, new_path
        
        # Create driver for this thread
        driver = create_driver()
        
        # Create search query and encode it for URL
        search_query = f"{company_profile} company profile"
        encoded_query = urllib.parse.quote_plus(search_query)
        
        # Navigate directly to DuckDuckGo search results
        duckduckgo_search_url = f"https://duckduckgo.com/?q={encoded_query}"
        driver.get(duckduckgo_search_url)
        
        # Reduced wait time
        WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='result']"))
        )
        
        # Quick cookie acceptance
        accept_cookies(driver)
        
        # Find first non-LinkedIn result (simplified)
        result_selectors = [
            "[data-testid='result'] h2 a[href]",
            "[data-testid='result'] a[href]:not([href*='duckduckgo.com'])"
        ]
        
        first_link = None
        for selector in result_selectors:
            try:
                results = driver.find_elements(By.CSS_SELECTOR, selector)
                for result in results[:5]:  # Check only first 5 results
                    href = result.get_attribute("href")
                    if (href and 
                        "duckduckgo.com" not in href and
                        not is_linkedin_page(href)):
                        first_link = href
                        break
                if first_link:
                    break
            except:
                continue
        
        if not first_link:
            print(f"[WARN] Thread {threading.current_thread().name}: No results for job_id={job_id}")
            return job_id, None
        
        # Navigate to first result
        driver.get(first_link)
        time.sleep(1)  # Minimal wait
        
        # Quick cookie acceptance
        accept_cookies(driver)
        
        # Quick scroll and screenshot
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        driver.execute_script("window.scrollTo(0, 0);")
        
        # Create label folder
        label_folder = os.path.join(OUTPUT_DIR, str(label))
        os.makedirs(label_folder, exist_ok=True)
        
        # Save screenshot
        image_path = os.path.join(label_folder, f"{job_id}.png")
        driver.save_screenshot(image_path)
        
        # Update cache (thread-safe)
        with cache_lock:
            company_cache[cache_key] = image_path
        
        print(f"[SUCCESS] Thread {threading.current_thread().name}: Screenshot saved for job_id={job_id}")
        return job_id, image_path

    except Exception as e:
        print(f"[ERROR] Thread {threading.current_thread().name}: job_id={job_id} - {e}")
        return job_id, None
    finally:
        if driver:
            driver.quit()

# =========================
# Main execution with parallel processing
# =========================
def main():
    # Load dataset (same as before)
    df = pd.read_csv("fake_job_postings.csv")
    
    # Data preprocessing (same as before)
    df['company_profile'] = df['company_profile'].fillna('')
    df['description'] = df['description'].fillna('')
    df['company_profile'] = df.apply(
        lambda row: row['company_profile'] if row['company_profile'].strip() != '' else row['description'],
        axis=1
    )
    df['company_profile'] = df['company_profile'].str[:50]
    
    # Handle command line arguments (same as before)
    start_job_id = None
    if len(sys.argv) > 1:
        try:
            start_job_id = int(sys.argv[1])
            start_index = df[df['job_id'] == start_job_id].index
            if len(start_index) == 0:
                print(f"[ERROR] Job ID {start_job_id} not found in dataset.")
                sys.exit(1)
            df = df.loc[start_index[0]:].reset_index(drop=True)
        except ValueError:
            print(f"[ERROR] Invalid job_id provided: {sys.argv[1]}. Must be an integer.")
            sys.exit(1)
    
    # Load cache
    company_cache = load_cache()
    
    print(f"[INFO] Processing {len(df)} jobs with {MAX_WORKERS} parallel workers")
    
    # Prepare job data for parallel processing
    job_data = [(row['job_id'], row['company_profile'], row['fraudulent']) for _, row in df.iterrows()]
    
    # Process jobs in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all jobs
        future_to_job = {executor.submit(process_single_job, job, company_cache): job[0] for job in job_data}
        
        # Process completed jobs
        for future in as_completed(future_to_job):
            job_id, image_path = future.result()
            results[job_id] = image_path
            
            # Periodic cache saving
            if len(results) % 10 == 0:
                save_cache(company_cache)
                print(f"[INFO] Processed {len(results)}/{len(job_data)} jobs")
    
    # Final cache save
    save_cache(company_cache)
    
    # Update dataframe with results
    df['image_path'] = df['job_id'].map(results)
    
    # Save results
    output_filename = f"fake_job_postings_with_images_from_{start_job_id}.csv" if start_job_id else "fake_job_postings_with_images.csv"
    df.to_csv(output_filename, index=False)
    
    print(f"[COMPLETE] All jobs processed! Results saved to: {output_filename}")
    print(f"[INFO] Cache contains {len(company_cache)} entries")

if __name__ == "__main__":
    main()