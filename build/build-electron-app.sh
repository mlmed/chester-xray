ELECTRON_OSX_URL=https://github.com/electron/electron/releases/download/v9.1.2/electron-v9.1.2-darwin-x64.zip
ELECTRON_WIN64_URL=https://github.com/electron/electron/releases/download/v9.1.2/electron-v9.1.2-linux-arm64.zip

mkdir -p ./cache
#wget -c $ELECTRON_OSX_URL -O ./cache/$(basename $ELECTRON_OSX_URL)
#wget -c $ELECTRON_WIN64_URL -O ./cache/$(basename $ELECTRON_WIN64_URL)


rm -rf ./tmp


mkdir -p ./tmp
mkdir -p ./tmp/app
mkdir -p ./tmp/app/res
mkdir -p ./tmp/app/models
mkdir -p ./tmp/app/examples

cp ../index.htm ./tmp/app/
cp ../system.js ./tmp/app/
cp ../style.css ./tmp/app/
cp -r ../res/ ./tmp/app/res/
cp -r ../examples/ ./tmp/app/examples/
cp -r ../models/ ./tmp/app/models/

cp -r ../models/xrv-all-45rot15trans15scale ./tmp/app/models/ 
cp -r ../models/chestxnet-45rot15trans15scale4byte ./tmp/app/models/ 


#mac app
unzip -q ./cache/$(basename $URL) -d ./tmp/

mv ./tmp/Electron.app ./tmp/Chester.app
rm ./tmp/Chester.app/Contents/Resources/electron.icns
cp Chester.icns ./tmp/Chester.app/Contents/Resources/
cp Info.plist ./tmp/Chester.app/Contents/
mkdir -p ./tmp/Chester.app/Contents/Resources/app/
cp -r ./tmp/app/ ./tmp/Chester.app/Contents/Resources/app/
cp app.js ./tmp/Chester.app/Contents/Resources/app/
cp package.json ./tmp/Chester.app/Contents/Resources/app/

#zip -r ./tmp/Chester.app.zip ./tmp/Chester.app

# windows app







