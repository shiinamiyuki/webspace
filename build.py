import os
shader = open('render.glsl','r')
src = shader.read() 
js = open('src.js')
src2 = js.read()
out = open('webgl.js','w')
out.write('const fsSource = `{0}`;\n{1}'.format(src,src2))
out.close()
os.system('"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" D:\Documents\Files\webspace\index.html')