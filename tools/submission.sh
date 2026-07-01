#!/usr/bin/env bash

function submission() {
  local root="$(git rev-parse --show-toplevel)"
  local submission_dir="$root/$1/submissions"
  mkdir -p "$submission_dir"
  cp "$root/$1/src/submission.csv" "$submission_dir/"
  echo "Submission file copied to $submission_dir/submission.csv"