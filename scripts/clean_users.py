#!/usr/bin/env python3
"""
Delete all IAM users whose name begins with 's3-scoped-'.
    • Requires: boto3 1.34+ and valid AWS credentials with iam:ListUsers,
      iam:DeleteUser, iam:DeleteAccessKey, etc.
    • Dry-run first!  Set DRY_RUN = True to see what would happen with no deletions.
"""

import boto3
from botocore.exceptions import ClientError

PREFIX = "s3-scoped-"
DRY_RUN = False  # change to False when you’re ready
iam = boto3.client("iam")


def safe_delete(func, **kwargs):  # type: ignore
    """Run an IAM API call and ignore ‘NoSuch*’ errors."""
    try:
        if not DRY_RUN:
            func(**kwargs)
    except ClientError as e:
        if e.response["Error"]["Code"].startswith(("NoSuch", "NoSuchEntity")):
            return
        raise


def delete_user(user_name: str):
    # 1. Delete access keys
    for k in iam.list_access_keys(UserName=user_name)["AccessKeyMetadata"]:
        safe_delete(
            iam.delete_access_key, UserName=user_name, AccessKeyId=k["AccessKeyId"]
        )

    # 2. Inline policies
    for pol in iam.list_user_policies(UserName=user_name)["PolicyNames"]:
        safe_delete(iam.delete_user_policy, UserName=user_name, PolicyName=pol)

    # 3. Attached managed policies
    for pol in iam.list_attached_user_policies(UserName=user_name)["AttachedPolicies"]:
        safe_delete(
            iam.detach_user_policy, UserName=user_name, PolicyArn=pol["PolicyArn"]
        )

    # 4. Groups
    for grp in iam.list_groups_for_user(UserName=user_name)["Groups"]:
        safe_delete(
            iam.remove_user_from_group, UserName=user_name, GroupName=grp["GroupName"]
        )

    # 5. MFA devices
    for mfa in iam.list_mfa_devices(UserName=user_name)["MFADevices"]:
        safe_delete(
            iam.deactivate_mfa_device,
            UserName=user_name,
            SerialNumber=mfa["SerialNumber"],
        )
        safe_delete(iam.delete_virtual_mfa_device, SerialNumber=mfa["SerialNumber"])

    # 6. Login profile
    safe_delete(iam.delete_login_profile, UserName=user_name)

    # 7. Signing certificates & SSH/SAML public keys
    for cert in iam.list_signing_certificates(UserName=user_name)["Certificates"]:
        safe_delete(
            iam.delete_signing_certificate,
            UserName=user_name,
            CertificateId=cert["CertificateId"],
        )
    for ssh in iam.list_ssh_public_keys(UserName=user_name)["SSHPublicKeys"]:
        safe_delete(
            iam.delete_ssh_public_key,
            UserName=user_name,
            SSHPublicKeyId=ssh["SSHPublicKeyId"],
        )
    for saml in iam.list_service_specific_credentials(UserName=user_name)[
        "ServiceSpecificCredentials"
    ]:
        safe_delete(
            iam.delete_service_specific_credential,
            UserName=user_name,
            ServiceSpecificCredentialId=saml["ServiceSpecificCredentialId"],
        )

    # 8. Finally delete the user
    safe_delete(iam.delete_user, UserName=user_name)
    action = "Would delete" if DRY_RUN else "Deleted"
    print(f"{action} user: {user_name}")


def main():
    paginator = iam.get_paginator("list_users")
    for page in paginator.paginate():
        for user in page["Users"]:
            name = user["UserName"]
            if name.startswith(PREFIX):
                delete_user(name)


if __name__ == "__main__":
    main()
