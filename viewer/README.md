# Daily validation viewer

## Brief installation instructions

1. Copy this directory tree to some directory outside of `DocumentRoot`, say,
`/var/local/daily_viewer/`.  Ensure it's all readable by `httpd`.

2. Symlink the `web` subdirectory into `DocumentRoot` as the desired URL path
(say `/daily/`):
```
  sudo ln -s /var/local/daily_viewer/web <DOCUMENT_ROOT>/daily
```

3. Apache config snippet to enable the script (again, `/daily/` is the URL path):
```
    WSGIApplicationGroup %{GLOBAL}
    WSGIScriptAlias /daily/view /var/local/daily_viewer/viewer.wsgi
    RedirectMatch ^/daily/$ /daily/view
```

4. Enable WSGI module:
```
sudo a2enmod wsgi
```

5. Restart apache
