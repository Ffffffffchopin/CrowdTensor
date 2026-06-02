#!/usr/bin/env node
import http from 'node:http'
import fs from 'node:fs/promises'
import { existsSync } from 'node:fs'
import path from 'node:path'
import { createHash, createHmac } from 'node:crypto'

import { createLibp2p } from 'libp2p'
import { tcp } from '@libp2p/tcp'
import { noise } from '@chainsafe/libp2p-noise'
import { yamux } from '@chainsafe/libp2p-yamux'
import { kadDHT } from '@libp2p/kad-dht'
import { identify } from '@libp2p/identify'
import { ping } from '@libp2p/ping'
import { generateKeyPair, privateKeyFromProtobuf, privateKeyToProtobuf } from '@libp2p/crypto/keys'
import { multiaddr } from '@multiformats/multiaddr'
import { fromString as uint8ArrayFromString } from 'uint8arrays/from-string'
import { toString as uint8ArrayToString } from 'uint8arrays/to-string'

if (typeof Promise.withResolvers !== 'function') {
  Object.defineProperty(Promise, 'withResolvers', {
    configurable: true,
    writable: true,
    value: () => {
      let resolve
      let reject
      const promise = new Promise((innerResolve, innerReject) => {
        resolve = innerResolve
        reject = innerReject
      })
      return { promise, resolve, reject }
    }
  })
}

const PROVIDER_RECORD_SCHEMA = 'real_p2p_provider_record_v1'
const PROVIDER_CATALOG_SCHEMA = 'real_p2p_provider_catalog_v1'
const ROUTE_LOOKUP_SCHEMA = 'real_p2p_route_lookup_v1'
const HEALTH_SCHEMA = 'real_p2p_health_v1'
const ANNOUNCE_SCHEMA = 'real_p2p_announce_v1'
const DIAGNOSTICS_SCHEMA = 'real_p2p_nat_relay_diagnostics_v1'
const LIBP2P_BACKEND_SCHEMA = 'libp2p_kad_backend_v1'
const PROVIDER_PROTOCOL = '/crowdtensor/provider-record/1.0.0'
const DEFAULT_KAD_PROTOCOL = '/crowdtensor/kad/1.0.0'

function parseArgs(argv) {
  const args = {
    host: '127.0.0.1',
    port: 8888,
    publicHost: '',
    swarmId: 'default',
    nodeId: '',
    role: 'observer',
    peerUrl: '',
    coordinatorUrl: '',
    backend: '',
    stageRole: '',
    stageCapability: [],
    capability: [],
    bootstrap: [],
    ttlSeconds: 60,
    recordSecret: '',
    requireSigned: false,
    signatureMaxAgeSeconds: 3600,
    discoveryBackend: 'libp2p-kad',
    printRecord: false,
    libp2pHost: '127.0.0.1',
    libp2pPort: 0,
    libp2pPublicHost: '',
    peerKeyFile: '',
    kadProtocol: DEFAULT_KAD_PROTOCOL
  }
  for (let i = 0; i < argv.length; i += 1) {
    const item = argv[i]
    const next = () => {
      i += 1
      if (i >= argv.length) {
        throw new Error(`${item} requires a value`)
      }
      return argv[i]
    }
    switch (item) {
      case '--host':
        args.host = next()
        break
      case '--port':
        args.port = Number(next())
        break
      case '--public-host':
        args.publicHost = next()
        break
      case '--swarm-id':
        args.swarmId = next()
        break
      case '--node-id':
        args.nodeId = next()
        break
      case '--role':
        args.role = next()
        break
      case '--peer-url':
        args.peerUrl = next()
        break
      case '--coordinator-url':
        args.coordinatorUrl = next()
        break
      case '--backend':
        args.backend = next()
        break
      case '--stage-role':
        args.stageRole = next()
        break
      case '--stage-capability':
        args.stageCapability.push(next())
        break
      case '--capability':
        args.capability.push(next())
        break
      case '--bootstrap':
        args.bootstrap.push(next())
        break
      case '--ttl-seconds':
        args.ttlSeconds = Number(next())
        break
      case '--record-secret':
      case '--peer-secret':
        args.recordSecret = next()
        break
      case '--require-signed':
        args.requireSigned = true
        break
      case '--signature-max-age-seconds':
        args.signatureMaxAgeSeconds = Number(next())
        break
      case '--discovery-backend':
        args.discoveryBackend = next()
        break
      case '--print-record':
        args.printRecord = true
        break
      case '--libp2p-host':
        args.libp2pHost = next()
        break
      case '--libp2p-port':
        args.libp2pPort = Number(next())
        break
      case '--libp2p-public-host':
        args.libp2pPublicHost = next()
        break
      case '--peer-key-file':
        args.peerKeyFile = next()
        break
      case '--kad-protocol':
        args.kadProtocol = next()
        break
      default:
        throw new Error(`unknown argument: ${item}`)
    }
  }
  if (!Number.isFinite(args.port) || args.port < 1) {
    throw new Error('--port must be positive')
  }
  if (!Number.isFinite(args.ttlSeconds) || args.ttlSeconds <= 0) {
    throw new Error('--ttl-seconds must be positive')
  }
  if (!Number.isFinite(args.signatureMaxAgeSeconds) || args.signatureMaxAgeSeconds <= 0) {
    throw new Error('--signature-max-age-seconds must be positive')
  }
  if (args.requireSigned && !args.recordSecret) {
    throw new Error('--require-signed requires --record-secret')
  }
  return args
}

function stableId(prefix, seed) {
  return `${prefix}-${createHash('sha256').update(String(seed || '')).digest('hex').slice(0, 16)}`
}

function hmacHash(value) {
  return createHash('sha256').update(String(value || '')).digest('hex')
}

function buildPeerIdentity(peerId, secret) {
  return {
    schema: 'p2p_lite_peer_identity_v1',
    identity_type: 'shared-secret-hmac',
    peer_id: String(peerId || ''),
    identity_hash: `sha256:${hmacHash(`crowdtensor-p2p-identity-v1\u0000${peerId}\u0000${secret}`)}`,
    decentralized_identity: false
  }
}

function stableJson(value) {
  if (Array.isArray(value)) {
    return `[${value.map(item => stableJson(item)).join(',')}]`
  }
  if (value && typeof value === 'object') {
    return `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${stableJson(value[key])}`).join(',')}}`
  }
  if (typeof value === 'number' && Number.isInteger(value)) {
    return `${value}.0`
  }
  return JSON.stringify(value)
}

function canonicalPeerForSignature(peer, signedAt) {
  const payload = {}
  for (const key of Object.keys(peer).sort()) {
    if (key === 'peer_signature' || key === 'health_score' || key === 'health_status' || key === 'identity_verified' || key === 'signature_verification' || key === 'peer_scoring' || key === 'trust_score' || key === 'trust_status') {
      continue
    }
    payload[key] = peer[key]
  }
  payload.signed_at = Number(signedAt)
  return stableJson(payload)
}

function signPeer(peer, secret, signedAt = Date.now() / 1000) {
  if (!secret) {
    return { ...peer }
  }
  const signed = { ...peer, peer_identity: buildPeerIdentity(peer.peer_id, secret) }
  const signature = createHmac('sha256', String(secret))
    .update(canonicalPeerForSignature(signed, signedAt))
    .digest('hex')
  signed.peer_signature = {
    schema: 'p2p_lite_signed_announce_v1',
    algorithm: 'hmac-sha256',
    identity_hash: signed.peer_identity.identity_hash,
    signed_at: Number(signedAt),
    signature
  }
  return signed
}

function verifyPeer(peer, secret, maxAgeSeconds) {
  const signature = typeof peer.peer_signature === 'object' && peer.peer_signature != null ? peer.peer_signature : {}
  const identity = typeof peer.peer_identity === 'object' && peer.peer_identity != null ? peer.peer_identity : {}
  if (!secret) {
    return { ok: false, diagnosis_code: 'p2p_peer_secret_missing' }
  }
  if (signature.schema !== 'p2p_lite_signed_announce_v1' || !['hmac-sha256', 'sha256-shared-secret'].includes(signature.algorithm)) {
    return { ok: false, diagnosis_code: 'p2p_signature_missing' }
  }
  const signedAt = Number(signature.signed_at || 0)
  if (signedAt <= 0) {
    return { ok: false, diagnosis_code: 'p2p_signature_time_missing' }
  }
  const now = Date.now() / 1000
  if (now - signedAt > Math.max(1, Number(maxAgeSeconds || 3600))) {
    return { ok: false, diagnosis_code: 'p2p_signature_expired' }
  }
  if (signedAt - now > 300) {
    return { ok: false, diagnosis_code: 'p2p_signature_from_future' }
  }
  const expectedIdentity = buildPeerIdentity(peer.peer_id, secret).identity_hash
  if (identity.identity_hash !== expectedIdentity || signature.identity_hash !== expectedIdentity) {
    return { ok: false, diagnosis_code: 'p2p_identity_hash_mismatch' }
  }
  const expected = signature.algorithm === 'hmac-sha256'
    ? createHmac('sha256', String(secret)).update(canonicalPeerForSignature(peer, signedAt)).digest('hex')
    : createHash('sha256').update(`${secret}\u0000${canonicalPeerForSignature(peer, signedAt)}`).digest('hex')
  if (signature.signature !== expected) {
    return { ok: false, diagnosis_code: 'p2p_signature_mismatch' }
  }
  return {
    ok: true,
    schema: 'p2p_lite_signature_verification_v1',
    diagnosis_code: 'p2p_signed_announce_verified',
    identity_hash: expectedIdentity,
    signed_at: signedAt
  }
}

function parseCapability(value) {
  if (!String(value || '').includes('=')) {
    return [String(value || '').trim(), 'true']
  }
  const [key, ...rest] = String(value).split('=')
  return [key.trim(), rest.join('=').trim()]
}

function sanitizePeer(value, args, now = Date.now() / 1000) {
  const caps = typeof value.capabilities === 'object' && value.capabilities != null ? { ...value.capabilities } : {}
  const urls = typeof value.urls === 'object' && value.urls != null ? { ...value.urls } : {}
  const ttl = Math.max(1, Math.min(Number(value.ttl_seconds || args.ttlSeconds || 60), 3600))
  const role = ['coordinator', 'miner', 'observer'].includes(String(value.role || '').toLowerCase()) ? String(value.role).toLowerCase() : 'observer'
  const peerId = String(value.peer_id || stableId('peer', JSON.stringify(value)))
  const peer = {
    schema: 'p2p_lite_peer_v1',
    swarm_id: String(value.swarm_id || args.swarmId || 'default'),
    peer_id: peerId,
    role,
    urls: {},
    capabilities: caps,
    stage_role: String(value.stage_role || caps.real_llm_sharded_stage_role || ''),
    backend: String(value.backend || caps.backend || ''),
    ttl_seconds: ttl,
    last_seen: Number(value.last_seen || now)
  }
  for (const key of ['coordinator', 'peer', 'metrics', 'health']) {
    if (urls[key]) {
      peer.urls[key] = String(urls[key])
    }
  }
  peer.expires_at = peer.last_seen + ttl
  if (typeof value.peer_identity === 'object' && value.peer_identity != null) {
    peer.peer_identity = value.peer_identity
  }
  if (typeof value.peer_signature === 'object' && value.peer_signature != null) {
    peer.peer_signature = value.peer_signature
  }
  return peer
}

function recordDigest(provider) {
  return createHash('sha256').update(JSON.stringify({
    swarm_id: provider.swarm_id,
    peer_id: provider.peer_id,
    role: provider.role,
    urls: provider.urls || {},
    capabilities: provider.capabilities || {},
    stage_role: provider.stage_role,
    backend: provider.backend,
    identity_hash: provider.peer_identity?.identity_hash || ''
  })).digest('hex')
}

function buildProviderRecord(peer, args, now = Date.now() / 1000) {
  const provider = sanitizePeer(peer, args, now)
  const caps = typeof provider.capabilities === 'object' && provider.capabilities != null ? provider.capabilities : {}
  const digest = recordDigest(provider)
  return {
    schema: PROVIDER_RECORD_SCHEMA,
    record_id: `provider-${digest.slice(0, 20)}`,
    swarm_id: provider.swarm_id,
    provider,
    role: provider.role,
    stage_role: provider.stage_role || caps.real_llm_sharded_stage_role || '',
    stage_capabilities: Array.isArray(caps.real_llm_sharded_stage_capabilities) ? caps.real_llm_sharded_stage_capabilities : [],
    backend: provider.backend || caps.backend || '',
    ttl_seconds: provider.ttl_seconds,
    last_seen: provider.last_seen,
    expires_at: provider.expires_at,
    signed_provider_record: Boolean(provider.peer_signature),
    signature_algorithm: provider.peer_signature?.algorithm || '',
    safety: {
      tokens_gossiped: false,
      raw_prompts_gossiped: false,
      activations_gossiped: false,
      peer_secret_gossiped: false,
      coordinator_backed_task_execution: true,
      not_large_model_serving: true
    }
  }
}

function providerFromRecord(recordOrPeer) {
  if (recordOrPeer?.schema === PROVIDER_RECORD_SCHEMA && typeof recordOrPeer.provider === 'object' && recordOrPeer.provider != null) {
    return { ...recordOrPeer.provider }
  }
  return { ...recordOrPeer }
}

function localPeerFromArgs(args) {
  const capabilities = {}
  for (const item of args.capability || []) {
    const [key, value] = parseCapability(item)
    if (!key) {
      continue
    }
    capabilities[key] = value.includes(',') ? value.split(',').map(part => part.trim()).filter(Boolean) : value
  }
  if (args.backend) {
    capabilities.backend = args.backend
  }
  if (args.stageRole) {
    capabilities.real_llm_sharded_stage_role = args.stageRole
  }
  if (args.stageCapability.length > 0) {
    capabilities.real_llm_sharded_stage_capabilities = [...args.stageCapability]
  }
  const urls = {}
  if (args.peerUrl) {
    urls.peer = args.peerUrl
  }
  if (args.coordinatorUrl) {
    urls.coordinator = args.coordinatorUrl
  }
  const peerId = args.nodeId || stableId('node', `${args.swarmId}:${args.role}:${args.peerUrl}:${args.coordinatorUrl}:${args.port}`)
  return {
    schema: 'p2p_lite_peer_v1',
    swarm_id: args.swarmId,
    peer_id: peerId,
    role: args.role,
    urls,
    capabilities,
    stage_role: args.stageRole,
    backend: args.backend,
    ttl_seconds: args.ttlSeconds
  }
}

async function loadOrCreatePrivateKey(filePath) {
  if (filePath && existsSync(filePath)) {
    const raw = JSON.parse(await fs.readFile(filePath, 'utf8'))
    return privateKeyFromProtobuf(Buffer.from(String(raw.private_key_protobuf_base64 || ''), 'base64'))
  }
  const key = await generateKeyPair('Ed25519')
  if (filePath) {
    await fs.mkdir(path.dirname(filePath), { recursive: true })
    await fs.writeFile(filePath, JSON.stringify({
      schema: 'crowdtensor_libp2p_peer_key_v1',
      key_type: 'Ed25519',
      private_key_protobuf_base64: Buffer.from(privateKeyToProtobuf(key)).toString('base64')
    }, null, 2) + '\n', { mode: 0o600 })
  }
  return key
}

function prune(records, now = Date.now() / 1000) {
  let removed = 0
  for (const [peerId, record] of records.entries()) {
    if (Number(record.expires_at || 0) <= now) {
      records.delete(peerId)
      removed += 1
    }
  }
  return removed
}

function providerCounts(peers) {
  let coordinator = 0
  let stage0 = 0
  let stage1 = 0
  for (const peer of peers) {
    if (peer.role === 'coordinator') {
      coordinator += 1
    }
    const caps = typeof peer.capabilities === 'object' && peer.capabilities != null ? peer.capabilities : {}
    const values = Array.isArray(caps.real_llm_sharded_stage_capabilities) ? caps.real_llm_sharded_stage_capabilities : []
    if (values.includes('real_llm_sharded_stage0') || values.includes('real_llm_sharded_cuda_stage0')) {
      stage0 += 1
    }
    if (values.includes('real_llm_sharded_stage1') || values.includes('real_llm_sharded_cuda_stage1')) {
      stage1 += 1
    }
  }
  return { coordinator, stage0, stage1 }
}

function metricInt(value) {
  const number = Number(value || 0)
  return Number.isFinite(number) ? Math.max(0, Math.trunc(number)) : 0
}

function providerResultMetrics(provider) {
  const caps = typeof provider.capabilities === 'object' && provider.capabilities != null ? provider.capabilities : {}
  const trust = typeof caps.trust === 'object' && caps.trust != null ? caps.trust : {}
  const metrics = typeof caps.peer_metrics === 'object' && caps.peer_metrics != null ? caps.peer_metrics : {}
  return {
    accepted_result_count: metricInt(caps.accepted_result_count ?? trust.accepted_result_count ?? metrics.accepted_result_count),
    failed_result_count: metricInt(caps.failed_result_count ?? trust.failed_result_count ?? metrics.failed_result_count),
    stale_result_count: metricInt(caps.stale_result_count ?? trust.stale_result_count ?? metrics.stale_result_count)
  }
}

function healthScore(provider, now = Date.now() / 1000) {
  const expiresAt = Number(provider.expires_at || 0)
  if (!Number.isFinite(expiresAt) || expiresAt <= now) {
    return 0
  }
  if (Number.isFinite(Number(provider.health_score))) {
    return Math.max(0, Math.min(100, Number(provider.health_score)))
  }
  return provider.identity_verified === true ? 100 : 90
}

function providerPeerScoring(provider, now = Date.now() / 1000) {
  const caps = typeof provider.capabilities === 'object' && provider.capabilities != null ? provider.capabilities : {}
  const trust = typeof caps.trust === 'object' && caps.trust != null ? caps.trust : {}
  const metrics = providerResultMetrics(provider)
  const heartbeatScore = Number(provider.health_score ?? healthScore(provider, now) ?? 0)
  const successBonus = Math.min(20, metrics.accepted_result_count * 2)
  const failurePenalty = Math.min(60, metrics.failed_result_count * 10 + metrics.stale_result_count * 5)
  const explicitQuarantine = Boolean(caps.quarantined || trust.quarantined)
  const score = Math.max(0, Math.min(100, heartbeatScore + successBonus - failurePenalty))
  const quarantined = Boolean(explicitQuarantine || score <= 0)
  return {
    schema: 'real_p2p_peer_scoring_v1',
    score: explicitQuarantine ? 0 : score,
    status: quarantined ? 'quarantined' : 'ready',
    quarantined,
    heartbeat_score: heartbeatScore,
    accepted_result_count: metrics.accepted_result_count,
    failed_result_count: metrics.failed_result_count,
    stale_result_count: metrics.stale_result_count,
    route_priority: quarantined ? 0 : score,
    diagnosis_codes: quarantined ? ['peer_quarantined', 'peer_scoring_ready'] : ['peer_scoring_ready']
  }
}

function peerSortKey(peer) {
  const scoring = typeof peer.peer_scoring === 'object' && peer.peer_scoring != null ? peer.peer_scoring : providerPeerScoring(peer)
  return [
    -Number(scoring.route_priority || 0),
    -Number(scoring.accepted_result_count || 0),
    String(peer.peer_id || '')
  ]
}

function peerScoringPayload(peers) {
  const now = Date.now() / 1000
  const ranked = peers.map(peer => {
    const scored = { ...peer }
    scored.peer_scoring = providerPeerScoring(scored, now)
    scored.trust_score = Number(scored.peer_scoring.score || 0)
    scored.trust_status = String(scored.peer_scoring.status || '')
    return scored
  }).sort((a, b) => {
    const left = peerSortKey(a)
    const right = peerSortKey(b)
    return left[0] - right[0] || left[1] - right[1] || left[2].localeCompare(right[2])
  })
  const scores = {}
  for (const peer of ranked) {
    const peerId = String(peer.peer_id || '')
    if (!peerId) {
      continue
    }
    scores[peerId] = peer.peer_scoring
  }
  return {
    schema: 'real_p2p_peer_scoring_v1',
    peer_count: ranked.length,
    quarantined_count: ranked.filter(peer => Boolean(peer.peer_scoring?.quarantined)).length,
    ranked_peer_ids: ranked.map(peer => String(peer.peer_id || '')),
    scores,
    diagnosis_codes: ['peer_scoring_ready']
  }
}

function catalogPayload({ args, records, node, bootstraps, providerSyncStats }) {
  prune(records)
  const providers = [...records.values()].map(record => {
    const provider = { ...(record.provider || {}) }
    provider.peer_scoring = providerPeerScoring(provider)
    provider.trust_score = Number(provider.peer_scoring.score || 0)
    provider.trust_status = String(provider.peer_scoring.status || '')
    return {
      ...record,
      provider,
      peer_scoring: provider.peer_scoring,
      trust_score: provider.trust_score,
      trust_status: provider.trust_status
    }
  }).sort((a, b) => String(a.record_id).localeCompare(String(b.record_id)))
  const peers = providers.map(record => record.provider).filter(Boolean).sort((a, b) => String(a.peer_id).localeCompare(String(b.peer_id)))
  const signedCount = providers.filter(record => record.identity_verified === true).length
  const healthyCount = providers.filter(record => Number(record.health_score || 0) > 0).length
  const peerScoring = peerScoringPayload(peers)
  return {
    schema: PROVIDER_CATALOG_SCHEMA,
    ok: true,
    swarm_id: args.swarmId,
    provider_count: providers.length,
    peer_count: peers.length,
    providers,
    peers,
    registry: {
      signed_provider_record_required: args.requireSigned,
      signed_provider_record_count: signedCount,
      healthy_provider_count: healthyCount,
      ttl_seconds: args.ttlSeconds,
      discovery_backend: 'libp2p-kad',
      provider_record_transport: 'libp2p-stream',
      kad_peer_routing_ready: true,
      peer_scoring_ready: true
    },
    peer_scoring: peerScoring,
    libp2p: libp2pStatus(args, node, bootstraps, providerSyncStats),
    diagnosis_codes: [
      'real_p2p_provider_store_ready',
      'replaceable_discovery_backend_ready',
      'libp2p_discovery_backend_ready',
      'p2p_peer_identity_ready',
      'p2p_provider_dht_ready',
      'peer_scoring_ready'
    ],
    safety: providerSafety(),
    boundaries: discoveryBoundaries()
  }
}

function providerSafety() {
  return {
    tokens_gossiped: false,
    raw_prompts_gossiped: false,
    activations_gossiped: false,
    peer_secret_gossiped: false,
    coordinator_backed_task_execution: true,
    not_large_model_serving: true
  }
}

function discoveryBoundaries() {
  return {
    replaceable_discovery_backend: true,
    discovery_backend: 'libp2p-kad',
    provider_record_store_ready: true,
    libp2p_runtime_ready: true,
    dht_runtime_ready: true,
    provider_record_transport: 'libp2p-stream',
    dht_provider_record_value_store_ready: false,
    nat_traversal_ready: false,
    relay_ready: false,
    production_p2p_security_ready: false,
    hivemind_petals_parity: false
  }
}

function libp2pStatus(args, node, bootstraps, providerSyncStats) {
  const multiaddrs = node.getMultiaddrs().map(addr => addr.toString())
  return {
    schema: LIBP2P_BACKEND_SCHEMA,
    ok: true,
    peer_id: node.peerId.toString(),
    stable_peer_identity: Boolean(args.peerKeyFile),
    listen_multiaddrs: multiaddrs,
    connected_peer_count: node.getPeers().length,
    connection_count: node.getConnections().length,
    bootstrap_peers: [...bootstraps],
    kad_protocol: args.kadProtocol,
    provider_protocol: PROVIDER_PROTOCOL,
    provider_record_transport: 'libp2p-stream',
    peer_routing_ready: true,
    provider_sync: providerSyncStats,
    diagnosis_codes: [
      'libp2p_discovery_backend_ready',
      'p2p_peer_identity_ready',
      'p2p_provider_dht_ready'
    ],
    boundaries: discoveryBoundaries()
  }
}

function readBody(request) {
  return new Promise((resolve, reject) => {
    const chunks = []
    request.on('data', chunk => chunks.push(chunk))
    request.on('end', () => {
      const raw = Buffer.concat(chunks).toString('utf8')
      if (!raw) {
        resolve({})
        return
      }
      try {
        resolve(JSON.parse(raw))
      } catch (error) {
        reject(error)
      }
    })
    request.on('error', reject)
  })
}

function sendJson(response, statusCode, payload) {
  const encoded = JSON.stringify(payload)
  response.writeHead(statusCode, {
    'content-type': 'application/json',
    'content-length': Buffer.byteLength(encoded)
  })
  response.end(encoded)
}

function routeLookup(sessionRequest, coordinatorUrl, peers, args) {
  const requirements = typeof sessionRequest.route_requirements === 'object' && sessionRequest.route_requirements != null ? sessionRequest.route_requirements : {}
  const required = Array.isArray(requirements.required_capabilities) ? requirements.required_capabilities.map(String) : []
  const matched = {}
  for (const capability of required) {
    const peer = peers.find(item => {
      const caps = typeof item.capabilities === 'object' && item.capabilities != null ? item.capabilities : {}
      const values = Array.isArray(caps.real_llm_sharded_stage_capabilities) ? caps.real_llm_sharded_stage_capabilities : []
      return values.includes(capability)
    })
    if (peer) {
      matched[capability] = String(peer.peer_id || '')
    }
  }
  let resolvedCoordinator = String(coordinatorUrl || '')
  if (!resolvedCoordinator) {
    const coord = peers.find(item => item.role === 'coordinator' && item.urls?.coordinator)
    if (coord) {
      resolvedCoordinator = String(coord.urls.coordinator)
    }
  }
  const missing = required.filter(capability => matched[capability] == null)
  const usable = Boolean(resolvedCoordinator) && missing.length === 0
  return {
    schema: ROUTE_LOOKUP_SCHEMA,
    ok: usable,
    swarm_id: args.swarmId,
    route: {
      schema: 'session_route_decision_v1',
      usable_now: usable,
      coordinator_url_present: Boolean(resolvedCoordinator),
      coordinator_url: resolvedCoordinator,
      route_source: requirements.route_source || 'real-p2p-discovery',
      backend: sessionRequest.backend,
      workload_type: sessionRequest.workload_type,
      required_capabilities: required,
      matched_capabilities: matched,
      missing_capabilities: missing,
      diagnosis_codes: usable ? ['session_route_ready'] : [resolvedCoordinator ? 'stage_capability_missing' : 'coordinator_route_missing']
    },
    provider_count: peers.length,
    diagnosis_codes: usable ? ['real_p2p_route_lookup_ready', 'libp2p_route_lookup_ready'] : ['real_p2p_route_lookup_blocked'],
    boundaries: discoveryBoundaries()
  }
}

function normalizeStreamChunk(chunk) {
  if (chunk instanceof Uint8Array) {
    return chunk
  }
  if (typeof chunk?.subarray === 'function') {
    return chunk.subarray()
  }
  return new Uint8Array(chunk)
}

async function readStreamJson(stream, maxBytes = 1048576) {
  let text = ''
  for await (const chunk of stream) {
    text += uint8ArrayToString(normalizeStreamChunk(chunk))
    if (text.length > maxBytes) {
      throw new Error('provider stream message too large')
    }
    if (text.includes('\n')) {
      break
    }
  }
  const line = text.split('\n')[0] || '{}'
  return JSON.parse(line)
}

async function writeStreamJson(stream, payload) {
  await stream.send(uint8ArrayFromString(`${JSON.stringify(payload)}\n`))
}

function mergeProviderRecord(records, recordOrPeer, args) {
  const now = Date.now() / 1000
  const peer = sanitizePeer(providerFromRecord(recordOrPeer), args, now)
  const verification = args.recordSecret ? verifyPeer(peer, args.recordSecret, args.signatureMaxAgeSeconds) : { ok: false, diagnosis_code: 'real_p2p_unsigned_provider_record' }
  if (args.requireSigned && !verification.ok) {
    throw new Error(`signed provider record required: ${verification.diagnosis_code}`)
  }
  peer.identity_verified = Boolean(verification.ok)
  peer.signature_verification = verification
  peer.health_score = Number(peer.expires_at || 0) > now ? (verification.ok ? 100 : 90) : 0
  peer.health_status = peer.health_score > 0 ? 'ready' : 'expired'
  peer.peer_scoring = providerPeerScoring(peer, now)
  peer.trust_score = Number(peer.peer_scoring.score || 0)
  peer.trust_status = String(peer.peer_scoring.status || '')
  const record = buildProviderRecord(peer, args, now)
  record.expires_at = Math.min(Number(record.expires_at || now + args.ttlSeconds), now + args.ttlSeconds)
  record.identity_verified = peer.identity_verified
  record.signature_verification = peer.signature_verification
  record.health_score = peer.health_score
  record.health_status = peer.health_status
  record.peer_scoring = peer.peer_scoring
  record.trust_score = peer.trust_score
  record.trust_status = peer.trust_status
  const existing = records.get(peer.peer_id)
  const existingLastSeen = Number(existing?.provider?.last_seen || 0)
  if (existing && existingLastSeen > Number(peer.last_seen || 0)) {
    return existing
  }
  records.set(peer.peer_id, record)
  return record
}

async function createNode(args) {
  const privateKey = await loadOrCreatePrivateKey(args.peerKeyFile)
  const listen = [`/ip4/${args.libp2pHost}/tcp/${Number(args.libp2pPort || 0)}`]
  const announceHost = args.libp2pPublicHost || args.publicHost
  const announce = announceHost && Number(args.libp2pPort || 0) > 0
    ? [`/ip4/${announceHost}/tcp/${Number(args.libp2pPort)}`]
    : []
  return createLibp2p({
    privateKey,
    addresses: { listen, announce },
    transports: [tcp()],
    connectionEncrypters: [noise()],
    streamMuxers: [yamux()],
    services: {
      identify: identify(),
      ping: ping(),
      dht: kadDHT({ protocol: args.kadProtocol, clientMode: false })
    }
  })
}

async function dialBootstrap(node, bootstraps, stats) {
  for (const item of bootstraps) {
    try {
      const addr = multiaddr(item)
      const conn = await node.dial(addr, { signal: AbortSignal.timeout(8000) })
      stats.bootstrap_dial_attempts += 1
      stats.bootstrap_dial_success += 1
      stats.last_bootstrap_peer = conn.remotePeer?.toString?.() || ''
      try {
        const found = await node.peerRouting.findPeer(conn.remotePeer, { signal: AbortSignal.timeout(8000) })
        if (found?.id) {
          stats.kad_find_peer_success += 1
        }
      } catch {
        stats.kad_find_peer_failed += 1
      }
    } catch (error) {
      stats.bootstrap_dial_attempts += 1
      stats.bootstrap_dial_failed += 1
      stats.last_error = `${error.name || 'Error'}: ${String(error.message || error).slice(0, 160)}`
    }
  }
}

async function syncWithPeer(node, peer, records, args, stats) {
  try {
    const stream = await node.dialProtocol(peer, PROVIDER_PROTOCOL, { signal: AbortSignal.timeout(8000) })
    await writeStreamJson(stream, {
      schema: 'crowdtensor_provider_sync_request_v1',
      type: 'catalog',
      swarm_id: args.swarmId,
      providers: [...records.values()]
    })
    const payload = await readStreamJson(stream)
    const incoming = Array.isArray(payload.providers) ? payload.providers : []
    let merged = 0
    for (const item of incoming) {
      try {
        mergeProviderRecord(records, item, args)
        merged += 1
      } catch {
        // ignore invalid remote records
      }
    }
    stats.provider_stream_sync_success += 1
    stats.provider_stream_records_merged += merged
    return true
  } catch (error) {
    stats.provider_stream_sync_failed += 1
    stats.last_error = `${error.name || 'Error'}: ${String(error.message || error).slice(0, 160)}`
    return false
  }
}

async function syncAllConnected(node, records, args, stats) {
  const peers = node.getPeers()
  for (const peer of peers) {
    await syncWithPeer(node, peer, records, args, stats)
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2))
  const records = new Map()
  const stats = {
    bootstrap_dial_attempts: 0,
    bootstrap_dial_success: 0,
    bootstrap_dial_failed: 0,
    kad_find_peer_success: 0,
    kad_find_peer_failed: 0,
    provider_stream_sync_success: 0,
    provider_stream_sync_failed: 0,
    provider_stream_records_merged: 0,
    inbound_provider_stream_requests: 0,
    last_bootstrap_peer: '',
    last_error: ''
  }
  const node = await createNode(args)
  await node.handle(PROVIDER_PROTOCOL, async (stream) => {
    try {
      stats.inbound_provider_stream_requests += 1
      const payload = await readStreamJson(stream)
      const incoming = Array.isArray(payload.providers) ? payload.providers : []
      for (const item of incoming) {
        try {
          mergeProviderRecord(records, item, args)
        } catch {
          // ignore invalid remote records
        }
      }
      await writeStreamJson(stream, {
        schema: 'crowdtensor_provider_sync_response_v1',
        ok: true,
        swarm_id: args.swarmId,
        providers: [...records.values()]
      })
    } catch (error) {
      await writeStreamJson(stream, { ok: false, error: error.name || 'Error', detail: String(error.message || error).slice(0, 160) }).catch(() => {})
    }
  })
  await node.start()

  const localPeer = sanitizePeer(localPeerFromArgs(args), args)
  const localRecord = buildProviderRecord(signPeer(localPeer, args.recordSecret), args)
  if (args.printRecord) {
    process.stdout.write(`${JSON.stringify(localRecord, null, 2)}\n`)
  }
  mergeProviderRecord(records, localRecord, args)

  await dialBootstrap(node, args.bootstrap, stats)
  await syncAllConnected(node, records, args, stats)
  const syncTimer = setInterval(() => {
    prune(records)
    syncAllConnected(node, records, args, stats).catch(error => {
      stats.last_error = `${error.name || 'Error'}: ${String(error.message || error).slice(0, 160)}`
    })
  }, 2000)

  const server = http.createServer(async (request, response) => {
    const url = new URL(request.url || '/', `http://${args.host}:${args.port}`)
    try {
      if (request.method === 'GET' && url.pathname === '/real-p2p/health') {
        const catalog = catalogPayload({ args, records, node, bootstraps: args.bootstrap, providerSyncStats: stats })
        sendJson(response, 200, {
          ok: true,
          schema: HEALTH_SCHEMA,
          swarm_id: args.swarmId,
          provider_count: catalog.provider_count,
          registry: catalog.registry,
          libp2p: catalog.libp2p,
          boundaries: catalog.boundaries
        })
        return
      }
      if (request.method === 'POST' && url.pathname === '/real-p2p/announce') {
        const payload = await readBody(request)
        const record = mergeProviderRecord(records, payload, args)
        await syncAllConnected(node, records, args, stats)
        sendJson(response, 200, { ok: true, schema: ANNOUNCE_SCHEMA, record, peer: record.provider, libp2p: libp2pStatus(args, node, args.bootstrap, stats) })
        return
      }
      if (request.method === 'GET' && url.pathname === '/real-p2p/providers') {
        await syncAllConnected(node, records, args, stats)
        sendJson(response, 200, catalogPayload({ args, records, node, bootstraps: args.bootstrap, providerSyncStats: stats }))
        return
      }
      if (request.method === 'POST' && url.pathname === '/real-p2p/route') {
        await syncAllConnected(node, records, args, stats)
        const payload = await readBody(request)
        const catalog = catalogPayload({ args, records, node, bootstraps: args.bootstrap, providerSyncStats: stats })
        sendJson(response, 200, routeLookup(payload.session_request || {}, String(payload.coordinator_url || ''), catalog.peers || [], args))
        return
      }
      if (request.method === 'GET' && url.pathname === '/real-p2p/diagnostics') {
        sendJson(response, 200, {
          schema: DIAGNOSTICS_SCHEMA,
          ok: true,
          bind_host: args.host,
          public_host: args.publicHost,
          listen_port: args.port,
          bootstrap_count: args.bootstrap.length,
          discovery_backend: 'libp2p-kad',
          libp2p: libp2pStatus(args, node, args.bootstrap, stats),
          nat_traversal_ready: false,
          relay_ready: false,
          operator_action: 'Expose libp2p TCP listen addresses directly, through VPN/tunnel, or through a future relay; automatic relay/NAT traversal is not enabled in this alpha.',
          diagnosis_codes: [
            'real_p2p_nat_relay_diagnostics_ready',
            args.bootstrap.length ? 'bootstrap_peer_configured' : 'bootstrap_peer_missing',
            'libp2p_discovery_backend_ready',
            'p2p_peer_identity_ready'
          ],
          boundaries: discoveryBoundaries()
        })
        return
      }
      if (request.method === 'GET' && url.pathname === '/peer/catalog') {
        const catalog = catalogPayload({ args, records, node, bootstraps: args.bootstrap, providerSyncStats: stats })
        sendJson(response, 200, {
          schema: 'p2p_lite_catalog_v1',
          ok: true,
          swarm_id: catalog.swarm_id,
          peer_count: catalog.peer_count,
          peers: catalog.peers,
          registry: catalog.registry,
          safety: catalog.safety,
          compatibility: { served_by: PROVIDER_CATALOG_SCHEMA }
        })
        return
      }
      if (request.method === 'POST' && url.pathname === '/peer/announce') {
        const payload = await readBody(request)
        const record = mergeProviderRecord(records, payload, args)
        await syncAllConnected(node, records, args, stats)
        sendJson(response, 200, { ok: true, schema: 'p2p_lite_announce_v1', peer: record.provider })
        return
      }
      sendJson(response, 404, { ok: false, error: 'not_found' })
    } catch (error) {
      sendJson(response, 422, { ok: false, error: error.name || 'Error', detail: String(error.message || error).slice(0, 240) })
    }
  })

  await new Promise((resolve) => server.listen(args.port, args.host, resolve))
  const stop = async () => {
    clearInterval(syncTimer)
    await new Promise(resolve => server.close(resolve))
    await node.stop()
  }
  process.on('SIGTERM', () => stop().then(() => process.exit(0)).catch(() => process.exit(1)))
  process.on('SIGINT', () => stop().then(() => process.exit(0)).catch(() => process.exit(1)))
}

main().catch(error => {
  process.stderr.write(`${error.stack || error.message || error}\n`)
  process.exit(1)
})
