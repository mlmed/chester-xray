ELECTRON_OSX_URL=https://github.com/electron/electron/releases/download/v9.1.2/electron-v9.1.2-darwin-x64.zip
ELECTRON_WIN64_URL=https://github.com/electron/electron/releases/download/v9.1.2/electron-v9.1.2-win32-x64.zip

mkdir -p ./cache
#wget -c $ELECTRON_OSX_URL -O ./cache/$(basename $ELECTRON_OSX_URL)
#wget -c $ELECTRON_WIN64_URL -O ./cache/$(basename $ELECTRON_WIN64_URL)


rm -rf ./tmp
rm -rf ./tmp

mkdir -p ./tmp
mkdir -p ./tmp/app
mkdir -p ./tmp/app/res
mkdir -p ./tmp/app/models
mkdir -p ./tmp/app/examples

cp ../index.htm ./tmp/app/
cp -r ../res/ ./tmp/app/res/
cp -r ../examples/ ./tmp/app/examples/

cp -r ../models/xrv-all-45rot15trans15scale ./tmp/app/models/ 
cp -r ../models/ae-chest-savedmodel-64-512 ./tmp/app/models/ 


#mac app
mkdir -p ./tmp/mac/
unzip -o -q ./cache/$(basename $ELECTRON_OSX_URL) -d ./tmp/mac/

mv ./tmp/mac/Electron.app ./tmp/mac/Chester.app
rm ./tmp/mac/Chester.app/Contents/Resources/electron.icns
cp Chester.icns ./tmp/mac/Chester.app/Contents/Resources/
cp Info.plist ./tmp/mac/Chester.app/Contents/
mkdir -p ./tmp/mac/Chester.app/Contents/Resources/app/
cp -r ./tmp/app/ ./tmp/mac/Chester.app/Contents/Resources/app/
cp app.js ./tmp/mac/Chester.app/Contents/Resources/app/
cp package.json ./tmp/mac/Chester.app/Contents/Resources/app/

# windows app
mkdir -p ./tmp/win/
unzip -o -q ./cache/$(basename $ELECTRON_WIN64_URL) -d ./tmp/win/

mkdir -p ./tmp/win/resources/app/
mv ./tmp/win/electron.exe ./tmp/win/chester.exe
rm ./tmp/win/resources/default_app.asar
cp -r ./tmp/app/ ./tmp/win/resources/app/
cp app.js ./tmp/win/resources/app/
cp package.json ./tmp/win/resources/app/





