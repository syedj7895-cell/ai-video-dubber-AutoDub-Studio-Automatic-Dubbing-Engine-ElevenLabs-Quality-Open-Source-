# Kilo Code Real-Time Visible Mode Implementation Plan

## Goal
Implement a toggleable real-time visible mode for Kilo Code that allows users to see file edits and terminal operations in real-time within VS Code, while maintaining backward compatibility with the existing background mode.

## Overview
This plan outlines the modifications needed to make Kilo Code behave like Cline in terms of visible real-time changes, with the ability to switch between modes based on user preference.

## Current Behavior (Background Mode)
- File edits: Direct file system modifications using available tools (edit, bash, etc.)
- Terminal operations: Commands execute in background without visible terminal output
- Changes appear in VS Code editor after a delay due to file watching

## Desired Real-Time Visible Mode Behavior
- File edits: Applied through VS Code's TextEditor API for instant visibility in open editors
- Terminal operations: Executed in visible VS Code terminal panels with real-time output
- Toggle mechanism: Configuration setting or command to switch between modes

## Implementation Details

### 1. Configuration System
Add a configuration setting to control the mode:
```json
// In kilo.json or VS Code settings
{
  "kiloCode.visibleMode": {
    "enabled": false, // Default to background mode for backward compatibility
    "fileEdits": true, // Apply file edits through VS Code API when true
    "terminalOps": true // Show terminal operations in visible panels when true
  }
}
```

### 2. File Edit Operations (Visible Mode)
When visible mode is enabled for file edits:
```typescript
// Instead of direct file system edits:
await vscode.workspace.fs.writeFile(uri, newContent);

// Use VS Code TextEditor API:
if (vscode.window.activeTextEditor && 
    vscode.window.activeTextEditor.document.uri.fsPath === uri.fsPath) {
  const editor = vscode.window.activeTextEditor;
  await editor.edit(editBuilder => {
    editBuilder.replace(new vscode.Range(0, 0, editor.document.lineCount, 0), newContent);
  });
} else {
  // Fallback to direct edit if file not open or prefer background
  await vscode.workspace.applyEdit(new vscode.WorkspaceEdit().set(uri, [
    new vscode.TextEdit(new vscode.Range(0, 0, new vscode.Position(
      await vscode.workspace.openTextDocument(uri).then(doc => doc.lineCount), 0)),
      newContent)
  ]));
}
```

### 3. Terminal Operations (Visible Mode)
When visible mode is enabled for terminal operations:
```typescript
// Instead of background bash execution:
const { stdout } = await cp.exec(command);

// Use VS Code Terminal API:
let terminal = vscode.window.terminals.find(t => t.name === "Kilo Code");
if (!terminal) {
  terminal = vscode.window.createTerminal("Kilo Code");
}
terminal.show(); // Make visible to user
terminal.sendText(`${command}\n`);

// For commands needing output capture, implement polling mechanism
// or use VS Code's terminal API to read output
```

### 4. Mode Switching Mechanism
Add commands to toggle modes:
- `kiloCode.toggleVisibleMode` - Toggle overall visible mode
- `kiloCode.setBackgroundMode` - Switch to background mode
- `kiloCode.setVisibleMode` - Switch to visible mode

### 5. Implementation Files to Modify
Based on typical Kilo Code structure (to be confirmed during implementation):
- `src/configuration.ts` - Add visible mode settings
- `src/fileOperations.ts` - Modify file edit logic
- `src/terminalOperations.ts` - Modify command execution logic
- `src/commands.ts` - Add mode switching commands
- `kilo.json` - Add default configuration

### 6. Backward Compatibility
- Default visible mode to `false` to maintain existing behavior
- When visible mode is disabled, fall back to current background implementation
- Ensure all existing functionality continues to work unchanged

### 7. Validation Approach
1. Test file edits in both modes:
   - Background: Verify file changes on disk, check VS Code update delay
   - Visible: Verify instant appearance in open editors
2. Test terminal operations in both modes:
   - Background: Verify command execution without visible terminal
   - Visible: Verify terminal appears with real-time output
3. Test mode switching:
   - Verify settings persist between sessions
   - Verify immediate effect when switching modes
4. Edge cases:
   - File not open in editor (visible mode should fallback gracefully)
   - Multiple file edits
   - Long-running terminal commands

## Risks and Mitigations

### Risk 1: Conflicts with User Editing
- **Problem**: VS Code API edits might conflict with user's active editing
- **Mitigation**: Check if editor has unsaved dirty state before applying edits; provide option to queue edits

### Risk 2: Performance Impact
- **Problem**: Using VS Code API for every edit might be slower than direct file access
- **Mitigation**: Batch multiple edits when possible; provide configuration to adjust behavior

### Risk 3: Terminal Focus Issues
- **Problem**: Automatically showing terminals might disrupt user workflow
- **Mitigation**: Make terminal visibility configurable; only show when explicitly needed for output

### Risk 4: Extension Host Complexity
- **Problem**: Adding VS Code API dependencies increases complexity
- **Mitigation**: Encapsulate VS Code-specific code behind interfaces; maintain clean separation

## Dependencies
- VS Code extension API (vscode module)
- TypeScript definitions for VS Code
- Existing Kilo Code infrastructure

## Open Questions for Implementation
1. What is the exact structure of Kilo Code's source code?
2. How are file operations currently implemented?
3. How are terminal/command operations currently implemented?
4. Where is configuration currently stored and accessed?
5. Are there existing patterns for feature flags/toggles in the codebase?

## Next Steps
1. Examine existing Kilo Code source structure
2. Implement configuration system for visible mode
3. Modify file operation handlers to conditionally use VS Code API
4. Modify terminal/command handlers to conditionally use visible terminals
5. Add mode switching commands and UI elements
6. Test thoroughly in both modes