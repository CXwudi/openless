import { useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Modal } from '../../components/ui/Modal';
import { Btn, Card } from '../_atoms';
import { ExperimentalSectionTitle, SettingRow, Toggle } from './shared';
import {
  acknowledgeExperimentalAsr,
  downloadPercent,
  deleteExperimentalAsr,
  experimentalAsrName,
  getExperimentalAsrStatus,
  needsModelGuidance,
  setExperimentalAsrEnabled,
  type ExperimentalAsrModel,
  type ExperimentalAsrStatus,
} from '../../lib/experimentalAsr';

// Owned by SettingsModal, not this subpage: leaving Experiments doesn't discard
// progress or miss the download-completed prompt. The native Host owns the jobs.
export function useExperimentalAsr(enabled: boolean) {
  const [models, setModels] = useState<ExperimentalAsrStatus[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState<ExperimentalAsrModel[]>([]);
  const alive = useRef(false);
  const sequence = useRef(0);
  const requests = useRef(new Set<ExperimentalAsrModel>());
  const channelSignature = models
    .map((s) => `${s.model}:${s.phase}:${s.channelId ?? ''}`)
    .join('|');
  useEffect(() => {
    window.dispatchEvent(new CustomEvent('ol-experimental-asr-channels-changed'));
  }, [channelSignature]);
  const refresh = useCallback(async () => {
    const request = ++sequence.current;
    try {
      const next = await getExperimentalAsrStatus();
      if (alive.current && request === sequence.current) {
        setModels(next);
        setError(null);
      }
    } catch (e) {
      if (alive.current && request === sequence.current) setError(String(e));
    }
  }, []);
  useEffect(() => {
    if (!enabled) return;
    alive.current = true;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      await refresh();
      if (alive.current) timer = setTimeout(() => void poll(), 1000);
    };
    void poll();
    return () => {
      alive.current = false;
      ++sequence.current;
      clearTimeout(timer);
    };
  }, [enabled, refresh]);
  const toggle = async (model: ExperimentalAsrModel, value: boolean) => {
    if (requests.current.has(model)) return;
    requests.current.add(model);
    setPending([...requests.current]);
    try {
      await setExperimentalAsrEnabled(model, value);
      await refresh();
    } catch (e) {
      if (alive.current) setError(String(e));
    } finally {
      requests.current.delete(model);
      if (alive.current) setPending([...requests.current]);
    }
  };
  const acknowledge = async (ready: ExperimentalAsrStatus[]) => {
    await Promise.all(ready.map((s) => acknowledgeExperimentalAsr(s.model)));
    await refresh();
  };
  const remove = async (model: ExperimentalAsrModel) => {
    if (requests.current.has(model)) return;
    requests.current.add(model);
    setPending([...requests.current]);
    try {
      await deleteExperimentalAsr(model);
      await refresh();
    } catch (error) {
      if (alive.current) setError(String(error));
    } finally {
      requests.current.delete(model);
      if (alive.current) setPending([...requests.current]);
    }
  };
  return { models, error, pending, toggle, acknowledge, refresh, remove };
}
export type ExperimentalAsrController = ReturnType<typeof useExperimentalAsr>;

export function ExperimentalAsrSection({
  controller,
  onOpenAsr,
}: {
  controller: ExperimentalAsrController;
  onOpenAsr: () => Promise<void>;
}) {
  const { t } = useTranslation();
  const { models, error, pending, toggle, remove } = controller;
  const [navigationError, setNavigationError] = useState<string | null>(null);
  return (
    <Card className="ol-experimental-models">
      <ExperimentalSectionTitle badge={t('common.experimental')}>
        {t('experimentalAsr.title')}
      </ExperimentalSectionTitle>
      {navigationError && <p role="alert">{navigationError}</p>}
      {error && (
        <p role="alert" style={{ color: 'var(--ol-warn)' }}>
          {error}
        </p>
      )}
      {models.length === 0 && <p role="status">{t('common.loading')}</p>}
      {models.map((status) => {
        const name = experimentalAsrName(status.model);
        const percent = downloadPercent(status);
        return (
          <section key={status.model} data-experimental-model={status.model}>
            <SettingRow
              label={
                status.model === 'text-stream'
                  ? `${name} · ${t('experimentalAsr.recommended')}`
                  : name
              }
              desc={
                <>
                  <p>
                    {t(
                      status.model === 'text-stream'
                        ? 'experimentalAsr.textStreamDescription'
                        : status.model === 'sense-voice'
                          ? 'experimentalAsr.senseVoiceDescription'
                          : 'experimentalAsr.gemmaDescription',
                    )}
                  </p>
                  <p>{t('experimentalAsr.description')}</p>
                  <p>{t('experimentalAsr.memoryHint')}</p>
                  {status.model === 'text-stream' && <p>{t('experimentalAsr.streamHint')}</p>}
                  <p>{t('experimentalAsr.deleteHint')}</p>
                  <p>{t('experimentalAsr.pipelineHint')}</p>
                  {!status.available && (
                    <p>
                      {t(`experimentalAsr.${status.unavailableReason ?? 'desktopRuntimeRequired'}`)}
                    </p>
                  )}
                </>
              }
            >
              <Toggle
                label={name}
                on={status.enabled}
                disabled={
                  pending.includes(status.model) ||
                  status.phase === 'removing' ||
                  (!status.available && !status.enabled)
                }
                onToggle={(next) => void toggle(status.model, next)}
              />
              <span
                role="status"
                aria-live="polite"
                className={
                  status.phase === 'disabled' || status.phase === 'ready'
                    ? 'ol-sr-only'
                    : 'ol-model-status'
                }
              >
                {t(`experimentalAsr.phases.${status.phase}`)}
                {status.phase === 'downloading' && ` · ${percent}%`}
              </span>
              {status.phase === 'failed' && (
                <Btn
                  size="sm"
                  variant="ghost"
                  onClick={() => void toggle(status.model, status.enabled)}
                  disabled={pending.includes(status.model)}
                >
                  {t('experimentalAsr.retry')}
                </Btn>
              )}
              {!status.enabled && status.downloadedBytes > 0 && (
                <Btn
                  size="sm"
                  variant="ghost"
                  disabled={pending.includes(status.model) || status.phase === 'removing'}
                  onClick={() => {
                    if (window.confirm(t('localAsr.deleteConfirm', { name })))
                      void remove(status.model);
                  }}
                >
                  {t('localAsr.delete')}
                </Btn>
              )}
              {status.phase === 'ready' && (
                <Btn
                  size="sm"
                  variant="ghost"
                  onClick={() => void onOpenAsr().catch((e) => setNavigationError(String(e)))}
                >
                  {t('experimentalAsr.openChannels')}
                </Btn>
              )}
            </SettingRow>
            {status.phase === 'downloading' && (
              <progress
                aria-label={t('experimentalAsr.downloadProgress', { name })}
                value={percent}
                max={100}
                style={{ width: '100%', accentColor: 'var(--ol-blue)' }}
              />
            )}
            {status.error && (
              <p
                role="alert"
                style={{ fontSize: 12, color: 'var(--ol-warn)', overflowWrap: 'anywhere' }}
              >
                {status.error}
              </p>
            )}
          </section>
        );
      })}
    </Card>
  );
}

export function ExperimentalAsrGuidance({
  controller,
  onOpenAsr,
}: {
  controller: ExperimentalAsrController;
  onOpenAsr: () => Promise<void>;
}) {
  const { t } = useTranslation();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const ready = controller.models.filter(needsModelGuidance);
  if (ready.length === 0) return null;
  const close = async (navigate: boolean) => {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      if (navigate) await onOpenAsr();
      await controller.acknowledge(ready);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <Modal
      zIndex={1000}
      labelledBy="experimental-asr-ready-title"
      onClose={() => void close(false)}
      width="min(460px, calc(100vw - 32px))"
    >
      <h2 id="experimental-asr-ready-title" style={{ fontSize: 17 }}>
        {t('experimentalAsr.readyTitle')}
      </h2>
      <p>{ready.map((s) => experimentalAsrName(s.model)).join(' / ')}</p>
      <p style={{ fontSize: 13, lineHeight: 1.7, color: 'var(--ol-ink-3)' }}>
        {t('experimentalAsr.readyBody')}
      </p>
      <p style={{ fontSize: 12, lineHeight: 1.6, color: 'var(--ol-ink-3)' }}>
        {t('experimentalAsr.pipelineHint')}
      </p>
      {error && <p role="alert">{error}</p>}
      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 20 }}>
        <Btn variant="ghost" disabled={busy} onClick={() => void close(false)}>
          {t('experimentalAsr.later')}
        </Btn>
        <Btn disabled={busy} onClick={() => void close(true)}>
          {t('experimentalAsr.confirm')}
        </Btn>
      </div>
    </Modal>
  );
}
