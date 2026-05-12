# Migration Plan

## Cutover checkpoints

1. compatibility stage keeps legacy routes functional
2. service extraction stage moves recommendation logic to `src/services`
3. data contract stage introduces schema versioning and experiment manifests
4. ranking stage enables multi stage pipeline controls
5. operations stage activates profile based deployment behavior
