import { useEffect, useRef, useState, type CSSProperties, type KeyboardEvent } from 'react';
import { Collapse } from './ui/Disclosure';
import { ChevronDown } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import {
  chordModifiersFromPressedCodes,
  formatComboParts,
  MODIFIER_CHORD_PRIMARY,
  modifiersFromPressedCodes,
} from '../lib/hotkey';
import {
  primaryFromKeyboardEvent,
  formatShortcutSaveError,
  shortcutFromMouseEvent,
} from '../lib/hotkeyRecorder';
import { KbdGroup } from './Kbd';
import { setShortcutRecordingActive, validateShortcutBinding } from '../lib/ipc';
import type { ShortcutBinding } from '../lib/types';

export function ShortcutRecorder({
  value,
  onSave,
  disabled = false,
  onDisable,
  disableLabel,
  disableDisabled = false,
  disableHint,
  onReset,
  resetLabel,
  comboOnly = false,
  sideSpecificModifiers = false,
  allowMacDictationKey = false,
  allowMouseButtons = false,
}: {
  value: ShortcutBinding | null;
  onSave: (binding: ShortcutBinding) => Promise<void>;
  disabled?: boolean;
  /** When provided, the menu's Disable is clickable (shortcuts that may be disabled, e.g. QA / style switch). */
  onDisable?: () => void | Promise<void>;
  disableLabel?: string;
  /** Grays out Disable (core shortcuts can't be disabled); pair with disableHint for the reason. */
  disableDisabled?: boolean;
  disableHint?: string;
  /** When provided, renders Reset in the menu — restores the shortcut's default binding. */
  onReset?: () => void | Promise<void>;
  resetLabel?: string;
  /** Combos only (modifier+key / function key); reject lone modifiers since global hotkeys can't register them. */
  comboOnly?: boolean;
  /** Dictation start/stop only: record side-specific modifiers like cmd-left / ctrl-right. */
  sideSpecificModifiers?: boolean;
  /** macOS dictation only: choose the dedicated key as the single trigger. */
  allowMacDictationKey?: boolean;
  /** Windows dictation only; other shortcut consumers cannot install mouse hooks. */
  allowMouseButtons?: boolean;
}) {
  const { t } = useTranslation();
  const [recording, setRecording] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const nativeSelected = allowMacDictationKey && value?.primary === 'MacDictationKey';
  const nativeError =
    error &&
    ['Permission', 'Busy', 'Unavailable', 'Changed'].find((kind) =>
      error.includes(`macDictationKey${kind}`),
    );
  const pendingModifier = useRef<ShortcutBinding | null>(null);
  const pendingTimer = useRef<number | null>(null);
  const pressedCodes = useRef<Set<string>>(new Set());
  const rootRef = useRef<HTMLDivElement | null>(null);

  // While the menu is open: Esc or clicking outside collapses it, so it can't linger.
  useEffect(() => {
    if (!menuOpen) return;
    const onKeyDown = (e: globalThis.KeyboardEvent) => {
      if (e.key === 'Escape' && !e.isComposing) {
        e.preventDefault();
        e.stopPropagation();
        setMenuOpen(false);
        rootRef.current?.querySelector<HTMLButtonElement>('[aria-expanded]')?.focus();
      }
    };
    const onPointerDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setMenuOpen(false);
    };
    // Capture Escape before the surrounding settings dialog handles it.
    window.addEventListener('keydown', onKeyDown, true);
    window.addEventListener('mousedown', onPointerDown);
    return () => {
      window.removeEventListener('keydown', onKeyDown, true);
      window.removeEventListener('mousedown', onPointerDown);
    };
  }, [menuOpen]);

  const clearPressedCodes = () => {
    pressedCodes.current.clear();
  };

  const clearPendingModifier = () => {
    if (pendingTimer.current !== null) {
      window.clearTimeout(pendingTimer.current);
      pendingTimer.current = null;
    }
    pendingModifier.current = null;
  };

  const resetRecordingState = () => {
    clearPendingModifier();
    clearPressedCodes();
  };

  useEffect(
    () => () => {
      resetRecordingState();
    },
    [],
  );

  useEffect(() => {
    if (!disabled || !recording) return;
    setRecording(false);
    resetRecordingState();
  }, [disabled, recording]);

  const finish = async (binding: ShortcutBinding) => {
    try {
      await validateShortcutBinding(binding);
      await onSave(binding);
      resetRecordingState();
      setRecording(false);
      setError(null);
    } catch (reason) {
      setError(formatShortcutSaveError(reason, t('settings.recording.shortcutSaveFailed')));
    }
  };

  // The browser doesn't deliver Fn keydown: subscribe to the Rust CGEventTap-forwarded
  // event before activating backend recording, so a Fn pressed right as recording
  // starts can't arrive before the listener is registered. Read finish via ref so the
  // effect doesn't re-register on finish identity changes.
  const finishRef = useRef(finish);
  finishRef.current = finish;
  useEffect(() => {
    if (!recording) return;
    let unlisten: (() => void) | undefined;
    let cancelled = false;
    void (async () => {
      try {
        const { listen } = await import('@tauri-apps/api/event');
        const handle = await listen('fn-shortcut-pressed', () => {
          if (cancelled) return;
          setMenuOpen(false);
          void finishRef.current({ primary: 'Fn', modifiers: [] });
        });
        if (cancelled) handle();
        else unlisten = handle;
      } catch (error) {
        console.warn('[shortcut] fn-shortcut-pressed listener failed', error);
      }
      if (cancelled) return;
      try {
        await setShortcutRecordingActive(true);
        if (cancelled) await setShortcutRecordingActive(false);
      } catch (error) {
        console.warn('[shortcut] recording state sync failed', error);
      }
    })();
    const onMouseDown = (e: MouseEvent) => {
      if (cancelled || !allowMouseButtons) return;
      const binding = shortcutFromMouseEvent(e);
      if (!binding) return;
      e.preventDefault();
      e.stopPropagation();
      clearPendingModifier();
      void finishRef.current(binding);
    };
    window.addEventListener('mousedown', onMouseDown, true);
    return () => {
      cancelled = true;
      unlisten?.();
      window.removeEventListener('mousedown', onMouseDown, true);
      void setShortcutRecordingActive(false);
    };
  }, [recording, allowMouseButtons]);

  /** Start recording and collapse the menu — Reset/Disable go away once "record shortcut" is pressed. */
  const startRecording = () => {
    if (disabled || recording) return;
    setMenuOpen(false);
    setError(null);
    resetRecordingState();
    setRecording(true);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (!recording || disabled) return;
    e.preventDefault();
    e.stopPropagation();
    if (e.key === 'Escape') {
      setRecording(false);
      setError(null);
      resetRecordingState();
      return;
    }
    if (isModifierKey(e.key)) {
      pressedCodes.current.add(e.code);
      if (comboOnly) {
        return;
      }
      if (sideSpecificModifiers) {
        const modifiers = chordModifiersFromPressedCodes(pressedCodes.current);
        if (modifiers.length >= 2) {
          clearPendingModifier();
          const binding = { primary: MODIFIER_CHORD_PRIMARY, modifiers };
          pendingModifier.current = binding;
          pendingTimer.current = window.setTimeout(() => {
            if (pendingModifier.current === binding) {
              void finish(binding);
            }
          }, 650);
          return;
        }
      }
      const primary = modifierPrimaryFromCode(e.code, e.key);
      if (!primary || pendingModifier.current?.primary === primary) return;
      clearPendingModifier();
      const binding = { primary, modifiers: [] };
      pendingModifier.current = binding;
      pendingTimer.current = window.setTimeout(() => {
        if (pendingModifier.current?.primary === primary) {
          void finish(binding);
        }
      }, 650);
      return;
    }
    clearPendingModifier();
    const primary = primaryFromKeyboardEvent(e);
    if (primary) {
      void finish({
        primary,
        modifiers: modifiersFromPressedCodes(pressedCodes.current, sideSpecificModifiers),
      });
    }
  };

  const onKeyUp = (e: KeyboardEvent<HTMLDivElement>) => {
    if (!recording || disabled || !isModifierKey(e.key)) return;
    e.preventDefault();
    e.stopPropagation();
    pressedCodes.current.delete(e.code);
    if (comboOnly) return;
    if (pendingModifier.current?.primary === MODIFIER_CHORD_PRIMARY) {
      const binding = pendingModifier.current;
      clearPendingModifier();
      void finish(binding);
      return;
    }
    const primary = modifierPrimaryFromCode(e.code, e.key);
    if (primary && pendingModifier.current?.primary === primary) {
      const binding = pendingModifier.current;
      clearPendingModifier();
      void finish(binding);
    }
  };

  const doReset = async () => {
    setMenuOpen(false);
    setError(null);
    try {
      await onReset?.();
    } catch (reason) {
      setError(formatShortcutSaveError(reason, t('settings.recording.shortcutSaveFailed')));
    }
  };

  const doDisable = () => {
    setMenuOpen(false);
    setError(null);
    if (onDisable) void onDisable();
  };

  // Disable is clickable: onDisable exists and it isn't grayed out (the dictation
  // shortcut is grayed; core hotkeys can't be disabled).
  const canDisable = Boolean(onDisable) && !disableDisabled;

  const rootStyle: CSSProperties = {
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
    width: '100%',
    // The shortcut recorder in settings rows no longer spans the full row (previously
    // the value hugged left and the chevron flew to the far edge); capped to the input
    // width, sitting compactly after the label column.
    maxWidth: 360,
  };
  const recorderRowStyle: CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    flexWrap: 'wrap',
    width: '100%',
  };
  const chevronButtonStyle: CSSProperties = {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: 26,
    height: 26,
    padding: 0,
    border: 0,
    borderRadius: 6,
    background: 'transparent',
    color: 'var(--ol-ink-4)',
    fontFamily: 'inherit',
    cursor: 'pointer',
    transition: 'background 0.16s var(--ol-motion-quick), color 0.16s var(--ol-motion-quick)',
  };
  const menuButtonStyle: CSSProperties = {
    fontSize: 12,
    padding: '5px 12px',
    border: '0.5px solid var(--ol-line-strong)',
    borderRadius: 6,
    background: 'transparent',
    color: 'var(--ol-ink-2)',
    fontFamily: 'inherit',
    fontWeight: 'var(--ol-font-weight-medium)',
    cursor: 'pointer',
    transition: 'background 0.16s var(--ol-motion-quick), color 0.16s var(--ol-motion-quick)',
  };
  const menuPrimaryStyle: CSSProperties = {
    ...menuButtonStyle,
    background: 'rgba(37,99,235,0.08)',
    borderColor: 'rgba(37,99,235,0.25)',
    color: 'var(--ol-blue)',
  };
  const disabledMenuButtonStyle: CSSProperties = {
    ...menuButtonStyle,
    opacity: 0.45,
    cursor: 'default',
  };

  return (
    <div style={rootStyle} className="ol-shortcut-recorder" ref={rootRef}>
      {recording ? (
        <div
          tabIndex={-1}
          onKeyDown={onKeyDown}
          onKeyUp={onKeyUp}
          ref={(element) => element?.focus()}
          className="ol-shortcut-recording"
        >
          {t('settings.recording.comboRecordHint')}
          <small>
            Esc · {t('common.cancel')}
            {allowMouseButtons && <> · {t('settings.recording.mouseSideHint')}</>}
          </small>
        </div>
      ) : (
        <>
          <div style={recorderRowStyle}>
            <button
              type="button"
              className="ol-shortcut-binding"
              disabled={disabled}
              onClick={startRecording}
              aria-label={t('settings.recording.comboRecordBtn')}
            >
              {value ? <KbdGroup keys={formatComboParts(value)} /> : <span>＋</span>}
            </button>
            <button
              type="button"
              disabled={disabled}
              onClick={() => setMenuOpen((open) => !open)}
              aria-label={t('settings.recording.comboMenuToggle', 'More options')}
              aria-expanded={menuOpen}
              style={chevronButtonStyle}
            >
              <ChevronDown
                size={14}
                style={{
                  transform: `rotate(${menuOpen ? 180 : 0}deg)`,
                  transition: 'transform 180ms var(--ol-motion-spring)',
                }}
              />
            </button>
          </div>
          <Collapse open={menuOpen}>
            <div className="ol-shortcut-menu">
              {allowMacDictationKey && (
                <button
                  type="button"
                  aria-pressed={nativeSelected}
                  disabled={disabled}
                  title={t('macDictationKey.description')}
                  style={nativeSelected ? menuPrimaryStyle : menuButtonStyle}
                  onClick={() => {
                    setMenuOpen(false);
                    void finish({ primary: 'MacDictationKey', modifiers: [] });
                  }}
                >
                  {t('macDictationKey.label')}
                </button>
              )}
              <button
                type="button"
                onClick={startRecording}
                disabled={disabled}
                style={menuPrimaryStyle}
              >
                {t('settings.recording.comboRecordBtn')}
              </button>
              {onReset && (
                <button type="button" onClick={doReset} disabled={disabled} style={menuButtonStyle}>
                  {resetLabel ?? t('settings.recording.comboResetBtn', 'Reset')}
                </button>
              )}
              <button
                type="button"
                onClick={doDisable}
                disabled={!canDisable}
                title={canDisable ? undefined : disableHint}
                style={canDisable ? menuButtonStyle : disabledMenuButtonStyle}
              >
                {disableLabel ?? t('settings.shortcuts.disable', 'Disable')}
              </button>
            </div>
          </Collapse>
        </>
      )}
      {error && (
        <div role="alert" style={{ fontSize: 12, color: 'var(--ol-err)' }}>
          {nativeError ? t(`macDictationKey.${nativeError}`) : error}
        </div>
      )}
    </div>
  );
}

function isModifierKey(key: string): boolean {
  return key === 'Control' || key === 'Alt' || key === 'Shift' || key === 'Meta';
}

function modifierPrimaryFromCode(code: string, key: string): string {
  if (code === 'ShiftLeft') return 'LeftShift';
  if (code === 'ShiftRight') return 'RightShift';
  if (code === 'ControlRight') return 'RightControl';
  if (code === 'ControlLeft') return 'LeftControl';
  if (code === 'AltRight') return 'RightOption';
  if (code === 'AltLeft') return 'LeftOption';
  if (code === 'MetaRight') return 'RightCommand';
  if (code === 'MetaLeft') return 'LeftCommand';
  if (key === 'Shift') return 'Shift';
  return '';
}
