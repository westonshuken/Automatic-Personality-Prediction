mdkir -p ~/.streamlit/

echo "\
[theme]
primaryColor='#943c3c'
backgroundColor='#e0e0e0'
secondaryBackgroundColor='#d8e4ff'
textColor='#461414'
font='monospace'
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml