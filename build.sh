# Add branch name to app title for feature branches
BRANCH=$(git branch --show-current)
SLUG="${BRANCH////-}"
if [[ "${SLUG}" != "master" ]]; then
  TITLE=Pictionary;
  BRANCH_TITLE="${SLUG}-${TITLE}"
  sed -i -e "s/${TITLE}/${BRANCH_TITLE}/g" buildozer.spec;
fi

# Phone needs to be plugged in with file transfer and usb debugging enabled
#buildozer -v android debug deploy run logcat | grep python

# Reset buildozer.spec
if [[ -n TITLE ]]; then
  sed -i -e "s/${BRANCH_TITLE}/${TITLE}/g" buildozer.spec;
fi
