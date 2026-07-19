# Project Website Deployment

The CrowdTensor website is packaged with the Volunteer Coordinator. The same
HTTPS origin serves:

- `/` for the project website;
- `/v1/volunteer/public-snapshot` for public Campaign progress;
- `/v1/volunteer/dashboard` for detailed public metrics;
- authenticated `/v1/volunteer/*` routes for contributor work.

This keeps promotional status and scheduling state consistent. The website
does not contain a manually edited "live" counter.

## Current Deployment

The founding deployment is available at
<https://crowdtensor.24.199.118.54.nip.io>. It runs a pinned
`HuggingFaceTB/SmolLM2-135M` and WikiText-2 LoRA Campaign with 100 target
rounds. Enrollment is controlled during beta; the 7B community Campaign shown
on the website remains in proposal phase.

The current workload proves and hardens the public contribution loop. It is not
presented as the final large-model Campaign or as evidence of useful model
quality improvement.

## Prepare A Campaign

Create a private Campaign directory outside the repository:

```bash
crowdtensor volunteer campaign import-smollm-wikitext \
  /var/lib/crowdtensor/campaigns/founding-smollm2-wikitext \
  --campaign-id crowdtensor-founding-smollm2-wikitext \
  --target-rounds 100 \
  --local-steps 1

crowdtensor volunteer campaign validate \
  /var/lib/crowdtensor/campaigns/founding-smollm2-wikitext --json
```

Keep `.private/` mode `0700` and invite files mode `0600`. Do not place the
Campaign directory, invite, model cache, tokenized data, or checkpoints in Git.

## Service Environment

Copy `deploy/site/crowdtensor-site.service.example` into systemd and create a
mode-0600 `/etc/crowdtensor/site.env`:

```text
CROWDTENSOR_REPO_DIR=/opt/crowdtensor
CROWDTENSOR_CAMPAIGN_DIR=/var/lib/crowdtensor/campaigns/founding-smollm2-wikitext
CROWDTENSOR_SITE_DOMAIN=train.example.org
CROWDTENSOR_SITE_PORT=8789
CROWDTENSOR_PROXY_ID=replace-with-a-private-random-value
CROWDTENSOR_PYTHON=/opt/crowdtensor/.venv/bin/python
```

The Coordinator binds only to `127.0.0.1`. Its TLS policy rejects direct HTTP
and trusts `X-Forwarded-Proto` only when the request carries the configured
loopback proxy identity.

## HTTPS Proxy

Run Caddy with `deploy/site/Caddyfile.example`, the same domain and proxy ID,
and persistent `/data` storage for certificate renewal. Ports 80 and 443 must
reach Caddy. Caddy obtains and renews the certificate and redirects HTTP to
HTTPS.

After startup, verify:

```bash
curl -I "https://${CROWDTENSOR_SITE_DOMAIN}/"
curl "https://${CROWDTENSOR_SITE_DOMAIN}/v1/volunteer/health"
curl "https://${CROWDTENSOR_SITE_DOMAIN}/v1/volunteer/public-snapshot"
```

Expected properties are HTTP 200, `tls_required=true`, the intended
`campaign_id`, and a public snapshot with no credentials, Cell identities, raw
data, token IDs, tensor values, or private paths.

## Opening Enrollment

Do not publish the invite. Review a contributor, transmit the mode-0600 invite
privately, and keep the public website limited to aggregate status. Before a
large 7B or pretraining Campaign opens, approve its model and dataset licenses,
evaluation suite, update bounds, moderation owner, rollback plan, and public
decision log through the Campaign proposal process.
