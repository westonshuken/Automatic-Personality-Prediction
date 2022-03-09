mkdir -p ~/.streamlit/

echo "\
[theme]
primaryColor='#ad5923'
backgroundColor='#c2c2d1'
secondaryBackgroundColor='#e6d2c5'
textColor='#371010'
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml