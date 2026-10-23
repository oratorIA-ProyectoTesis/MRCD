# Return to a campaign without losing progress

Campaign participants authenticate with a personal access token, not with their email address. Existing tokens and saved annotations remain valid. The server never rotates or reveals an existing token because someone knows an email.

## Participant quick path

1. **First visit:** open the campaign link, enter name/email, and explicitly accept the displayed version of the consent statement.
2. **Save the token privately:** the registration screen shows it once and offers a copy action. Treat it like a password. Continue after saving it.
3. **Same browser:** the browser retains the token until sign-out; revisit the campaign to continue the same tasks.
4. **Different device or after sign-out:** choose **Already participated?** and paste the saved personal token. It is validated without rotation, so another legitimate browser session is not revoked.
5. **Lost token:** contact the campaign administrator through the team's existing trusted channel. There is no automatic recovery email and email alone is not proof of identity.

Signing out clears this application's browser storage, including its token and local drafts. Do not share tokens or put them in URLs, screenshots, public logs, or annotation notes.

## Administrator recovery

Use **Administration → campaign participant → Reset access** only after manually checking the participant's identity through an existing trusted channel. Confirm that check and enter a brief audit reason, without credentials or sensitive identity documents. Deliver the replacement token privately to that verified participant.

Recovery preserves account ID, consent record, tasks, annotations, and progress. It revokes the previous token. It cannot reactivate an inactive participant, reset an unrelated campaign account, or change roles. Only an active administrator can perform it. The user's revision history records the administrator, time, reason, and token hash, not the plaintext replacement.

## API contract

| Operation | Authorization / behavior |
| --- | --- |
| `GET /api/campaigns/{slug}` | Public description plus current `consent_version` and exact `consent_text`. No participant credentials. |
| `POST /api/campaigns/{slug}/join` | New registrations only: name, email, strict `consent: true`, and current `consent_version`. Existing email → `409`, no account/session mutation. Closed/unavailable campaign → `404`. |
| `POST /api/campaigns/{slug}/resume` | Existing token in the `Authorization: Bearer …` header; no email or token in request URL/body. Invalid/inactive token → `401`; another campaign or role → `403`. Returns campaign/name, never a new token. |
| `POST /api/admin/campaigns/{slug}/participants/{id}/reset-access` | Administrator token, `identity_verified: true`, and a 10–500 character audit reason. Returns the replacement once. |

Credential-bearing responses use `Cache-Control: no-store`. Returning to a closed campaign may display existing tasks; it never registers a new participant or generates missing tasks. Closing admission is not a general editing lock on already assigned tasks.

## Integrity and compatibility

- Registration, capacity checks, and initial tasks share one database transaction serialized on the campaign row. Concurrent registration cannot exceed capacity or create duplicate users/tasks.
- Campaign extension and open-campaign return use the same lock and preserve existing tasks. New task IDs are deterministic per campaign/person/region; historical IDs remain unchanged.
- Campaign-only accounts cannot take global development-pool tasks, read unrelated tasks, open adjudication cases, or bypass task clips through example-recording endpoints. Ordinary non-campaign annotators retain their development-pool workflow.
- Generic assignment (`/admin/tasks`, `/review/tasks`, and the shared batch operation) rejects campaign accounts before creating any tasks. The generic selector excludes them; use the authorized campaign extension workflow for more campaign regions. Campaign progress ignores historical unscoped/other-campaign tasks instead of counting invisible work.
- Direct and automatic adjudication-case producers also reject campaign-only accounts before creating a case or consuming a pair; adjudicators must have an eligible active global role, independently of A/B.
- New consent records include version, UTC acceptance time, campaign, and the exact statement's SHA-256. Historical accounts lacking this record remain **consent evidence unavailable**; returning or resetting never invents a prior acceptance.
- Older clients that do not send the displayed consent version must reload/update before registering. Existing stored tokens require no migration or forced logout.
- Historical duplicate participants or conflicting annotations are not automatically merged. They need explicit integrity review; this security change does not rewrite reference data.

## Verification boundaries

Security regressions use synthetic participants and temporary SQLite/media. They cover unauthorized recovery, token possession, legacy consent, reset authorization/audit, rollback, concurrent capacity/identity/extension, and campaign boundaries. They do not establish participant identity in the real world, lawful corpus permission, human annotation quality, production PostgreSQL concurrency, or model accuracy. Browser interaction and deployment hardening remain separate integration checks in the closure plan.
