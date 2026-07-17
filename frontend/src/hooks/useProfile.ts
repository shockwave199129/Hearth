/** Re-exports shared profile types + hook from ProfileContext so existing
 * imports (`hooks/useProfile`) keep working. State lives in one provider
 * so App gating and Onboarding share the same active profile. */
export {
  ProfileProvider,
  useProfile,
  type OnboardingPayload,
  type Profile,
  type ProfileContextValue,
} from "../lib/ProfileContext";
