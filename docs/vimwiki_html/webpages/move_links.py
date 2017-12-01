from bs4 import BeautifulSoup
import os


cwd = os.getcwd()
all_files = os.listdir(cwd)
dirs = [cwd + '/' + d for d in all_files if os.path.isdir(d)]
html_files = []
for d in dirs:
    PATH = d + '/'
    files = [PATH + f for f in os.listdir(d) if f.endswith('.html')]
    html_files.extend([f for f in files if len(files) > 0])

for html_file in html_files:
    # Check each html file
    with open(html_file) as html:
        soup = BeautifulSoup(html, 'html.parser')
        base = soup.find('base')
        base['href'] = ''

        # Fix image links
        images = soup.find_all('img')
        for image in images:
            source = image['src']
            if source.endswith('.png'):
                source = source.split('/')[-1]
                image['src'] = source

        # Fix html links
        links = soup.find_all('a')
        for link in links:
            try:
                href = link['href']
                if href.endswith('.html'):
                    href = href.split('/')[-1]
                    link['href'] = href
            except:
                pass

        # Use the custom gwsumm.min.js
        scripts = soup.find_all('script')
        for script in scripts:
            if "gwsumm.min.js" in script['src']:
                script['src'] = "../gwsumm.min.js"

        home_nav_bar = False
        if "Home" in str(soup):
            home_nav_bar = True

        if not home_nav_bar:
            home = soup.find('ul', {'class':'nav navbar-nav'})
            home_link = '\n<li>\n<a href="../webpages.html">Home</a>\n</li>\n'
            home.insert(0, BeautifulSoup(home_link, 'html.parser'))

    with open(html_file, "wb") as f:
        f.write(soup.prettify("utf-8"))
