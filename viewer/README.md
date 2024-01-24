# Daily validation viewer

## Brief installation instructions

1. Create a virtualenv and install everything in `requirements.txt` to it.

2. Deploy the viewer to the virtualenv by using the `./deploy_to_venv` script.

3. Symlink the deployed `web` subdirectory into `DocumentRoot` as the desired
URL path (say `/daily/`), e.g.:
```
  sudo ln -s /opt/venvs/daily_viewer/viewer/web /srv/bao/daily
```

4. Create and populate the `rendered` subdirectory in the deployed viewer
directory (e.g. `/opt/venvs/daily_viewer/viewer/rendered`).  This may be a
symlink to a directory elsewehere.

5. Set up the reverse proxy in apache:
```
  ProxyPass /daily/view http://127.0.0.1:4884/
  ProxyPassReverse /daily/view http://127.0.0.1:4884/
  RedirectMatch ^/daily/$ /daily/view

  ProxyPass /daily-test/view http://127.0.0.1:4885/
  ProxyPassReverse /daily-test/view http://127.0.0.1:4885/
  RedirectMatch ^/daily-test/$ /daily-test/view
```

6. Start the gunicorn server (this does not need to be done from within the
virtualenv):
```
    /opt/venvs/daily_viewer/bin/gunicorn -b 127.0.0.1:4884 -w 4 /opt/venvs/daily_viewer/viewer/daily_viewer:application
```

## Test mode

You can run a test-server directly out of the current dirctory by
using the `./test_server` script _from within_ the virtualenv created in
step one above.
