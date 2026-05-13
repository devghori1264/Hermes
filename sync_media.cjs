const fs = require('fs')
const path = require('path')
const root = __dirname
const srcDir = path.join(root, 'media')
const pubMedia = path.join(root, 'public', 'media')
const pubIcon = path.join(root, 'public', 'main_icon.png')
if (!fs.existsSync(srcDir)) {
  process.stderr.write('sync_media: media directory missing\n')
  process.exit(1)
}
fs.mkdirSync(pubMedia, { recursive: true })
let copied = 0
for (const name of fs.readdirSync(srcDir)) {
  const from = path.join(srcDir, name)
  if (fs.statSync(from).isFile()) {
    fs.copyFileSync(from, path.join(pubMedia, name))
    copied += 1
  }
}
if (copied === 0) {
  process.stderr.write('sync_media: no files in media\n')
  process.exit(1)
}
const iconSrc = path.join(srcDir, 'main_icon.png')
if (fs.existsSync(iconSrc)) {
  fs.copyFileSync(iconSrc, pubIcon)
}
