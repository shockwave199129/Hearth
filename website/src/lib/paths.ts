// BASE_URL is "/Hearth/" in dev but "/Hearth" in a production build, so naive
// template concatenation silently produces "/Hearthbrand/logo.png".
const root = import.meta.env.BASE_URL.replace(/\/?$/, "/");

export const asset = (path: string) => `${root}${path.replace(/^\//, "")}`;

export const home = root;
